//! Reproducible, single-node benchmark runner for Vivvy.
//!
//! It separates complete durable-MemoryStore measurements from synthetic ANN
//! checks. Synthetic vectors validate mechanics, never semantic quality.

use rand::{rngs::StdRng, Rng, SeedableRng};
use std::collections::HashMap;
use std::fmt::Write as _;
use std::fs;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tempfile::tempdir;
use vivvy_core::{concurrent::VivvyIndex, distance::Metric, flat::FlatIndex};
use vivvy_memory::{
    ContextFormatOptions, ForgetRequest, MemoryConfig, MemoryFilter, MemoryKind, MemoryScope,
    MemoryStore, RecallRequest, RememberRequest,
};

const BENCHMARK_MODEL: &str = "synthetic-deterministic-v1";
const RNG_SEED: u64 = 0x56_49_56_56_59;

#[derive(Debug, Clone)]
struct Config {
    profile: String,
    records: usize,
    dimensions: usize,
    tenants: usize,
    queries: usize,
    warmup_queries: usize,
    top_k: usize,
    delete_percent: usize,
    report_path: PathBuf,
}

impl Config {
    fn for_profile(name: &str) -> Result<Self, String> {
        let (records, dimensions, tenants, queries, warmup_queries) = match name {
            "smoke" => (1_000, 384, 10, 200, 20),
            "standard" => (10_000, 768, 50, 1_000, 100),
            "large" => (100_000, 768, 100, 2_000, 200),
            _ => {
                return Err(format!(
                    "unknown profile '{name}'; use smoke, standard, or large"
                ))
            }
        };
        Ok(Self {
            profile: name.to_string(),
            records,
            dimensions,
            tenants,
            queries,
            warmup_queries,
            top_k: 10,
            delete_percent: 10,
            report_path: PathBuf::from(format!("benchmarks/reports/{name}.md")),
        })
    }

    fn validate(&self) -> Result<(), String> {
        if self.records == 0 || self.dimensions == 0 || self.tenants == 0 || self.queries == 0 {
            return Err(
                "records, dimensions, tenants, and queries must be greater than zero".into(),
            );
        }
        if self.top_k == 0 || self.delete_percent > 100 {
            return Err("top-k must be positive and delete-percent must be in 0..=100".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct LatencySummary {
    samples: usize,
    mean_ms: f64,
    min_ms: f64,
    p50_ms: f64,
    p95_ms: f64,
    p99_ms: f64,
    max_ms: f64,
}

impl LatencySummary {
    fn from_durations(durations: &[Duration]) -> Self {
        assert!(!durations.is_empty(), "benchmark recorded no samples");
        let mut ms: Vec<f64> = durations
            .iter()
            .map(|d| d.as_secs_f64() * 1_000.0)
            .collect();
        ms.sort_by(|a, b| a.total_cmp(b));
        let percentile = |p: f64| {
            let index = ((p * ms.len() as f64).ceil() as usize)
                .saturating_sub(1)
                .min(ms.len() - 1);
            ms[index]
        };
        Self {
            samples: ms.len(),
            mean_ms: ms.iter().sum::<f64>() / ms.len() as f64,
            min_ms: ms[0],
            p50_ms: percentile(0.50),
            p95_ms: percentile(0.95),
            p99_ms: percentile(0.99),
            max_ms: *ms.last().expect("non-empty after assertion"),
        }
    }
}

#[derive(Debug)]
struct Scenario {
    name: &'static str,
    operations: usize,
    elapsed: Duration,
    latency: Option<LatencySummary>,
    correctness: String,
    notes: String,
}

impl Scenario {
    fn throughput(&self) -> f64 {
        self.operations as f64 / self.elapsed.as_secs_f64()
    }
}

#[derive(Debug)]
struct RunResult {
    config: Config,
    started_at: String,
    host: HostInfo,
    scenarios: Vec<Scenario>,
    db_bytes_before_vacuum: u64,
    db_bytes_after_vacuum: u64,
    wal_bytes_before_vacuum: u64,
    wal_bytes_after_vacuum: u64,
}

#[derive(Debug)]
struct HostInfo {
    os: String,
    kernel: String,
    cpu_model: String,
    logical_cpus: usize,
    memory_kib: Option<u64>,
    rustc: String,
}

#[derive(Debug, Clone)]
struct FixtureRecord {
    scope: MemoryScope,
    embedding: Vec<f32>,
    content: String,
    kind: MemoryKind,
    id: Option<String>,
}

fn main() {
    let config = parse_args().unwrap_or_else(|error| fail(&format!("{error}\n\n{}", usage())));
    config.validate().unwrap_or_else(|error| fail(&error));
    println!("Vivvy production-oriented benchmark ({})", config.profile);
    println!(
        "records={} dims={} tenants={} queries={} top_k={}",
        config.records, config.dimensions, config.tenants, config.queries, config.top_k
    );

    let result = run(&config).unwrap_or_else(|error| fail(&format!("benchmark failed: {error}")));
    let report = render_report(&result);
    if let Some(parent) = config.report_path.parent() {
        fs::create_dir_all(parent).expect("create report directory");
    }
    fs::write(&config.report_path, report).expect("write benchmark report");
    for scenario in &result.scenarios {
        let latency = scenario
            .latency
            .as_ref()
            .map(|l| format!(" p95={:.3} ms", l.p95_ms))
            .unwrap_or_default();
        println!(
            "{:<34} {:>10.1} ops/s{}",
            scenario.name,
            scenario.throughput(),
            latency
        );
    }
    println!("report written to {}", config.report_path.display());
}

fn fail(message: &str) -> ! {
    eprintln!("error: {message}");
    std::process::exit(2)
}

fn parse_args() -> Result<Config, String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!("{}", usage());
        std::process::exit(0);
    }
    let profile = value_after(&args, "--profile").unwrap_or_else(|| "standard".to_string());
    let mut config = Config::for_profile(&profile)?;
    for (flag, target) in [
        ("--records", &mut config.records),
        ("--dimensions", &mut config.dimensions),
        ("--tenants", &mut config.tenants),
        ("--queries", &mut config.queries),
        ("--warmup-queries", &mut config.warmup_queries),
        ("--top-k", &mut config.top_k),
        ("--delete-percent", &mut config.delete_percent),
    ] {
        if let Some(value) = value_after(&args, flag) {
            *target = value
                .parse()
                .map_err(|_| format!("{flag} requires an integer, got '{value}'"))?;
        }
    }
    if let Some(value) = value_after(&args, "--report") {
        config.report_path = PathBuf::from(value);
    }
    Ok(config)
}

fn value_after(args: &[String], flag: &str) -> Option<String> {
    args.windows(2)
        .find(|pair| pair[0] == flag)
        .map(|pair| pair[1].clone())
}

fn usage() -> &'static str {
    "Usage: cargo run --release --package bench -- [options]\n\
     Profiles: smoke (1K/384d), standard (10K/768d; default), large (100K/768d)\n\
     Options: --profile <name> --records <N> --dimensions <N> --tenants <N>\n\
     --queries <N> --warmup-queries <N> --top-k <N> --delete-percent <0..100>\n\
     --report <path>"
}

fn run(config: &Config) -> Result<RunResult, String> {
    stage("synthetic ANN oracle and HNSW measurement");
    let mut scenarios = vec![run_synthetic_ann(config)?];
    let temp = tempdir().map_err(|e| format!("create temporary benchmark store: {e}"))?;
    let store_config = MemoryConfig::builder(temp.path())
        .dimensions(config.dimensions)
        .embedding_model(BENCHMARK_MODEL)
        .max_recall_limit(config.top_k.max(100))
        .build()
        .map_err(display_error("build store config"))?;
    stage("cold open");
    let cold_start = Instant::now();
    let mut store =
        MemoryStore::open(store_config.clone()).map_err(display_error("open fresh store"))?;
    let cold_elapsed = cold_start.elapsed();
    scenarios.push(one_operation(
        "cold start (empty store)",
        cold_elapsed,
        "store opened",
        "Includes SQLite schema/WAL initialization and empty index creation.",
    ));

    let mut fixtures = generate_fixture(config);
    stage("durable single-record ingestion");
    scenarios.push(load_store(&store, &mut fixtures)?);
    assert_health(&store, config.records, 0, "post-load")?;

    drop(store);
    stage("warm open and index rebuild");
    let warm_start = Instant::now();
    store = MemoryStore::open(store_config).map_err(display_error("warm reopen"))?;
    let warm_elapsed = warm_start.elapsed();
    assert_health(&store, config.records, 0, "warm-start")?;
    scenarios.push(one_operation(
        "warm start + index rebuild",
        warm_elapsed,
        &format!("{} active records rebuilt", config.records),
        "SQLite is canonical; opening rebuilds the ephemeral vector accelerator.",
    ));

    stage("scoped vector recall");
    scenarios.push(run_recall_scenario(
        "vector recall (scoped)",
        &store,
        &fixtures,
        config,
        RecallMode::Vector,
    )?);
    stage("hybrid recall and MMR");
    scenarios.push(run_recall_scenario(
        "hybrid recall + MMR (scoped)",
        &store,
        &fixtures,
        config,
        RecallMode::HybridMmr,
    )?);
    stage("filtered hybrid recall");
    scenarios.push(run_recall_scenario(
        "hybrid filtered recall (scope + kind)",
        &store,
        &fixtures,
        config,
        RecallMode::FilteredHybrid,
    )?);
    stage("context formatting");
    scenarios.push(run_context_scenario(&store, &fixtures, config)?);
    stage("online backup and restore");
    scenarios.push(run_backup_restore_scenario(&store, config)?);

    let before_vacuum = store
        .health()
        .map_err(display_error("read pre-vacuum health"))?;
    let delete_count = config.records * config.delete_percent / 100;
    stage("logical tombstones");
    scenarios.push(run_delete_scenario(&store, &fixtures, delete_count)?);
    stage("physical vacuum and index rebuild");
    scenarios.push(run_vacuum_scenario(&store, delete_count)?);
    let after_vacuum = store
        .health()
        .map_err(display_error("read post-vacuum health"))?;
    if after_vacuum.total_tombstoned_records != 0
        || after_vacuum.total_active_records != config.records.saturating_sub(delete_count)
        || !after_vacuum.is_healthy
    {
        return Err(format!(
            "post-vacuum invariant failed: active={} tombstones={} healthy={}",
            after_vacuum.total_active_records,
            after_vacuum.total_tombstoned_records,
            after_vacuum.is_healthy
        ));
    }
    Ok(RunResult {
        config: config.clone(),
        started_at: unix_timestamp(),
        host: collect_host_info(),
        scenarios,
        db_bytes_before_vacuum: before_vacuum.db_size_bytes,
        db_bytes_after_vacuum: after_vacuum.db_size_bytes,
        wal_bytes_before_vacuum: before_vacuum.wal_size_bytes,
        wal_bytes_after_vacuum: after_vacuum.wal_size_bytes,
    })
}

fn stage(name: &str) {
    println!("[stage] {name}");
}

fn one_operation(
    name: &'static str,
    elapsed: Duration,
    correctness: &str,
    notes: &str,
) -> Scenario {
    Scenario {
        name,
        operations: 1,
        elapsed,
        latency: Some(LatencySummary::from_durations(&[elapsed])),
        correctness: correctness.into(),
        notes: notes.into(),
    }
}

fn assert_health(
    store: &MemoryStore,
    active: usize,
    tombstones: usize,
    phase: &str,
) -> Result<(), String> {
    let health = store.health().map_err(display_error("read store health"))?;
    if health.total_active_records != active
        || health.total_tombstoned_records != tombstones
        || !health.is_healthy
    {
        return Err(format!(
            "{phase} health invariant failed: active={} tombstones={} healthy={}",
            health.total_active_records, health.total_tombstoned_records, health.is_healthy
        ));
    }
    Ok(())
}

fn display_error(context: &'static str) -> impl FnOnce(vivvy_memory::MemoryError) -> String {
    move |error| format!("{context}: {error}")
}

fn generate_fixture(config: &Config) -> Vec<FixtureRecord> {
    let mut rng = StdRng::seed_from_u64(RNG_SEED);
    (0..config.records).map(|record_no| {
        let tenant_no = record_no % config.tenants;
        let topic_no = record_no % 97;
        FixtureRecord {
            scope: MemoryScope::new(format!("tenant-{tenant_no:04}"), "agent-memory").expect("valid fixture scope"),
            embedding: (0..config.dimensions).map(|_| rng.random::<f32>()).collect(),
            content: format!("tenant-{tenant_no:04} agent memory topic-{topic_no:03} record-{record_no:08} durable benchmark payload"),
            kind: if record_no % 2 == 0 { MemoryKind::Fact } else { MemoryKind::Preference },
            id: None,
        }
    }).collect()
}

fn load_store(store: &MemoryStore, fixtures: &mut [FixtureRecord]) -> Result<Scenario, String> {
    let record_count = fixtures.len();
    let start = Instant::now();
    let mut samples = Vec::with_capacity(record_count);
    for record in fixtures.iter_mut() {
        let operation_start = Instant::now();
        let id = store
            .remember(RememberRequest {
                operation_id: None,
                scope: record.scope.clone(),
                content: record.content.clone(),
                embedding: record.embedding.clone(),
                kind: record.kind,
                importance: 0.8,
                expires_at_ms: None,
                metadata: HashMap::new(),
                source: HashMap::new(),
            })
            .map_err(display_error("insert memory"))?;
        samples.push(operation_start.elapsed());
        record.id = Some(id);
    }
    Ok(Scenario { name: "durable single-record writes", operations: record_count, elapsed: start.elapsed(), latency: Some(LatencySummary::from_durations(&samples)), correctness: format!("{record_count} IDs persisted"), notes: "Each operation includes SQLite persistence, FTS maintenance, operation journal, and HNSW insertion.".into() })
}

#[derive(Clone, Copy)]
enum RecallMode {
    Vector,
    HybridMmr,
    FilteredHybrid,
}

fn run_recall_scenario(
    name: &'static str,
    store: &MemoryStore,
    fixtures: &[FixtureRecord],
    config: &Config,
    mode: RecallMode,
) -> Result<Scenario, String> {
    let fixture_at = |i| &fixtures[(i * 7919) % fixtures.len()];
    for i in 0..config.warmup_queries {
        black_box(run_one_recall(store, fixture_at(i), config.top_k, mode)?);
    }
    let start = Instant::now();
    let mut samples = Vec::with_capacity(config.queries);
    let mut hits = 0usize;
    let mut scope_leaks = 0usize;
    for i in 0..config.queries {
        let fixture = fixture_at(i + config.warmup_queries);
        let operation_start = Instant::now();
        let response = run_one_recall(store, fixture, config.top_k, mode)?;
        samples.push(operation_start.elapsed());
        let expected = fixture.id.as_deref().expect("fixture persisted");
        hits += usize::from(response.items.iter().any(|item| item.memory.id == expected));
        scope_leaks += response
            .items
            .iter()
            .filter(|item| item.memory.scope != fixture.scope)
            .count();
    }
    if scope_leaks != 0 {
        return Err(format!(
            "scope isolation violation in {name}: {scope_leaks} foreign results"
        ));
    }
    let notes = match mode {
        RecallMode::Vector => "Dense HNSW candidate recall scoped by MemoryScope.",
        RecallMode::HybridMmr => "FTS5 + dense HNSW + RRF + default scoring + MMR(lambda=0.70).",
        RecallMode::FilteredHybrid => "Hybrid recall with tenant scope and MemoryKind predicate.",
    };
    Ok(Scenario {
        name,
        operations: config.queries,
        elapsed: start.elapsed(),
        latency: Some(LatencySummary::from_durations(&samples)),
        correctness: format!(
            "synthetic target Recall@{}={:.2}%; cross-scope results=0",
            config.top_k,
            100.0 * hits as f64 / config.queries as f64
        ),
        notes: notes.into(),
    })
}

fn run_one_recall(
    store: &MemoryStore,
    fixture: &FixtureRecord,
    top_k: usize,
    mode: RecallMode,
) -> Result<vivvy_memory::RecallResponse, String> {
    let (query_text, filters, mmr_lambda) = match mode {
        RecallMode::Vector => (None, MemoryFilter::default(), None),
        RecallMode::HybridMmr => (
            Some(fixture.content.clone()),
            MemoryFilter::default(),
            Some(0.70),
        ),
        RecallMode::FilteredHybrid => (
            Some(fixture.content.clone()),
            MemoryFilter::kind_in([fixture.kind]),
            Some(0.70),
        ),
    };
    store
        .recall(RecallRequest {
            scope: fixture.scope.clone(),
            query_embedding: fixture.embedding.clone(),
            query_text,
            limit: top_k,
            filters,
            include_explanations: false,
            mmr_lambda,
            scoring: Default::default(),
        })
        .map_err(display_error("recall"))
}

fn run_context_scenario(
    store: &MemoryStore,
    fixtures: &[FixtureRecord],
    config: &Config,
) -> Result<Scenario, String> {
    let options = ContextFormatOptions {
        max_tokens: 512,
        ..Default::default()
    };
    let count = config.queries.min(100);
    let start = Instant::now();
    let mut samples = Vec::with_capacity(count);
    for i in 0..count {
        let fixture = &fixtures[(i * 3571) % fixtures.len()];
        let operation_start = Instant::now();
        let output = store
            .format_context(
                RecallRequest {
                    scope: fixture.scope.clone(),
                    query_embedding: fixture.embedding.clone(),
                    query_text: Some(fixture.content.clone()),
                    limit: config.top_k,
                    filters: MemoryFilter::default(),
                    include_explanations: false,
                    mmr_lambda: Some(0.70),
                    scoring: Default::default(),
                },
                options.clone(),
            )
            .map_err(display_error("format context"))?;
        samples.push(operation_start.elapsed());
        if output.len() > options.max_tokens * 4 + 4 {
            return Err("format_context exceeded its char/4 token-budget estimator".into());
        }
    }
    Ok(Scenario { name: "token-budgeted context formatting", operations: count, elapsed: start.elapsed(), latency: Some(LatencySummary::from_durations(&samples)), correctness: format!("{count} outputs respected 512-token estimator budget"), notes: "Includes hybrid recall and formatting; token counting is the implementation's char/4 approximation.".into() })
}

fn run_backup_restore_scenario(store: &MemoryStore, config: &Config) -> Result<Scenario, String> {
    let backup_dir = tempdir().map_err(|e| format!("create backup directory: {e}"))?;
    let start = Instant::now();
    store
        .backup(backup_dir.path())
        .map_err(display_error("online backup"))?;
    let restore_start = Instant::now();
    let restored = MemoryStore::open(
        MemoryConfig::builder(backup_dir.path())
            .dimensions(config.dimensions)
            .embedding_model(BENCHMARK_MODEL)
            .max_recall_limit(config.top_k.max(100))
            .build()
            .map_err(display_error("build restore config"))?,
    )
    .map_err(display_error("open restored backup"))?;
    let restore_elapsed = restore_start.elapsed();
    assert_health(&restored, config.records, 0, "backup/restore")?;
    Ok(Scenario { name: "online backup + restore open", operations: config.records, elapsed: start.elapsed(), latency: Some(LatencySummary::from_durations(&[restore_elapsed])), correctness: format!("restored {} active records", config.records), notes: "Elapsed includes SQLite VACUUM INTO backup and restore index rebuild; latency columns represent restore-open.".into() })
}

fn run_delete_scenario(
    store: &MemoryStore,
    fixtures: &[FixtureRecord],
    count: usize,
) -> Result<Scenario, String> {
    if count == 0 {
        return Ok(Scenario {
            name: "tombstone writes",
            operations: 0,
            elapsed: Duration::from_secs(1),
            latency: None,
            correctness: "skipped (delete-percent=0)".into(),
            notes: String::new(),
        });
    }
    let start = Instant::now();
    let mut samples = Vec::with_capacity(count);
    for fixture in fixtures.iter().take(count) {
        let operation_start = Instant::now();
        store
            .forget(ForgetRequest {
                operation_id: None,
                scope: fixture.scope.clone(),
                id: fixture.id.clone().expect("fixture persisted"),
            })
            .map_err(display_error("tombstone memory"))?;
        samples.push(operation_start.elapsed());
    }
    Ok(Scenario {
        name: "tombstone writes",
        operations: count,
        elapsed: start.elapsed(),
        latency: Some(LatencySummary::from_durations(&samples)),
        correctness: format!("{count} records hidden from reads"),
        notes: "Logical delete; physical reclamation is measured by vacuum.".into(),
    })
}

fn run_vacuum_scenario(store: &MemoryStore, expected: usize) -> Result<Scenario, String> {
    let start = Instant::now();
    let purged = store
        .vacuum_tombstones(expected.max(1))
        .map_err(display_error("vacuum tombstones"))?;
    let elapsed = start.elapsed();
    if purged != expected {
        return Err(format!("vacuum purged {purged}; expected {expected}"));
    }
    Ok(Scenario {
        name: "tombstone vacuum + index rebuild",
        operations: purged,
        elapsed,
        latency: Some(LatencySummary::from_durations(&[elapsed])),
        correctness: format!("{purged} tombstones physically purged"),
        notes: "Physical deletion triggers a derived-vector-index rebuild in this implementation."
            .into(),
    })
}

fn run_synthetic_ann(config: &Config) -> Result<Scenario, String> {
    let count = config.records.min(20_000);
    let queries = config.queries.min(200);
    let mut rng = StdRng::seed_from_u64(RNG_SEED ^ 0xA11CE);
    let vectors: Vec<Vec<f32>> = (0..count)
        .map(|_| {
            (0..config.dimensions)
                .map(|_| rng.random::<f32>())
                .collect()
        })
        .collect();
    let query_vectors: Vec<Vec<f32>> = (0..queries)
        .map(|_| {
            (0..config.dimensions)
                .map(|_| rng.random::<f32>())
                .collect()
        })
        .collect();
    let mut flat = FlatIndex::new(Metric::L2);
    for (n, vector) in vectors.iter().enumerate() {
        flat.insert((n + 1) as u64, vector.clone());
    }
    let oracle: Vec<Vec<u64>> = query_vectors
        .iter()
        .map(|q| {
            flat.search(q, config.top_k)
                .into_iter()
                .map(|(id, _)| id)
                .collect()
        })
        .collect();
    let build_start = Instant::now();
    let index = VivvyIndex::new(config.dimensions, Metric::L2, None::<&str>, None::<&str>)
        .map_err(|e| format!("create ANN index: {e}"))?;
    for vector in vectors {
        index
            .insert(vector)
            .map_err(|e| format!("insert ANN vector: {e}"))?;
    }
    let build_elapsed = build_start.elapsed();
    for q in query_vectors.iter().take(10) {
        black_box(
            index
                .search(q, config.top_k)
                .map_err(|e| format!("warm ANN search: {e}"))?,
        );
    }
    let start = Instant::now();
    let mut samples = Vec::with_capacity(queries);
    let mut hits = 0usize;
    for (q, expected) in query_vectors.iter().zip(&oracle) {
        let operation_start = Instant::now();
        let result = index
            .search(q, config.top_k)
            .map_err(|e| format!("ANN search: {e}"))?;
        samples.push(operation_start.elapsed());
        hits += expected
            .iter()
            .filter(|id| result.iter().any(|(found, _)| found == *id))
            .count();
    }
    Ok(Scenario { name: "synthetic core ANN recall", operations: queries, elapsed: start.elapsed(), latency: Some(LatencySummary::from_durations(&samples)), correctness: format!("exact-oracle Recall@{}={:.2}%", config.top_k, 100.0 * hits as f64 / (queries * config.top_k) as f64), notes: format!("Random L2 vectors (N={count}); build {:.3}s ({:.1} inserts/s). Validates ANN mechanics only, not semantic quality.", build_elapsed.as_secs_f64(), count as f64 / build_elapsed.as_secs_f64()) })
}

fn collect_host_info() -> HostInfo {
    let cpuinfo = fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
    let cpu_model = cpuinfo
        .lines()
        .find_map(|line| {
            line.strip_prefix("model name\t: ")
                .or_else(|| line.strip_prefix("Hardware\t: "))
        })
        .unwrap_or("unavailable")
        .into();
    let memory_kib = fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|contents| {
            contents.lines().find_map(|line| {
                line.strip_prefix("MemTotal:")
                    .and_then(|value| value.split_whitespace().next())
                    .and_then(|value| value.parse().ok())
            })
        });
    HostInfo {
        os: std::env::consts::OS.into(),
        kernel: fs::read_to_string("/proc/sys/kernel/osrelease")
            .unwrap_or_else(|_| "unavailable".into())
            .trim()
            .into(),
        cpu_model,
        logical_cpus: std::thread::available_parallelism().map_or(1, usize::from),
        memory_kib,
        rustc: command_output("rustc", &["--version"]),
    }
}

fn command_output(program: &str, args: &[&str]) -> String {
    std::process::Command::new(program)
        .args(args)
        .output()
        .ok()
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .unwrap_or_else(|| "unavailable".into())
        .trim()
        .into()
}

fn unix_timestamp() -> String {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs().to_string())
        .unwrap_or_else(|_| "unavailable".into())
}

fn git_revision() -> String {
    let revision = command_output("git", &["rev-parse", "--short", "HEAD"]);
    let is_dirty = std::process::Command::new("git")
        .args(["status", "--porcelain", "--untracked-files=normal"])
        .output()
        .ok()
        .is_some_and(|output| !output.stdout.is_empty());
    if is_dirty {
        format!("{revision}-dirty")
    } else {
        revision
    }
}

fn render_report(result: &RunResult) -> String {
    let mut out = String::new();
    writeln!(out, "# Vivvy benchmark report\n").unwrap();
    writeln!(out, "- Run timestamp (Unix seconds): `{}`\n- Profile: `{}`\n- Git revision: `{}`\n- Command: `cargo run --release --package bench -- --profile {}`\n", result.started_at, result.config.profile, git_revision(), result.config.profile).unwrap();
    writeln!(out, "## Workload declaration\n\n| Field | Value |\n| --- | ---: |\n| Records | {} |\n| Embedding dimensions | {} |\n| Embedding model label | `{BENCHMARK_MODEL}` |\n| Tenants / namespace | {} / `agent-memory` |\n| Measured queries / warm-up queries | {} / {} |\n| Recall top-k | {} |\n| Delete rate before vacuum | {}% |\n| RNG seed | `0x{RNG_SEED:X}` |\n", result.config.records, result.config.dimensions, result.config.tenants, result.config.queries, result.config.warmup_queries, result.config.top_k, result.config.delete_percent).unwrap();
    writeln!(out, "## Environment\n\n| Field | Value |\n| --- | --- |\n| OS / kernel | {} / {} |\n| CPU | {} |\n| Logical CPUs | {} |\n| Memory | {} KiB |\n| Rust | {} |\n", result.host.os, result.host.kernel, result.host.cpu_model, result.host.logical_cpus, result.host.memory_kib.map_or_else(|| "unavailable".into(), |v| v.to_string()), result.host.rustc).unwrap();
    writeln!(out, "## Results\n\n| Scenario | Ops/s | Mean ms | p50 ms | p95 ms | p99 ms | Correctness |\n| --- | ---: | ---: | ---: | ---: | ---: | --- |").unwrap();
    for s in &result.scenarios {
        let values = s.latency.as_ref().map_or_else(
            || ["—".into(), "—".into(), "—".into(), "—".into()],
            |l| {
                [
                    format!("{:.3}", l.mean_ms),
                    format!("{:.3}", l.p50_ms),
                    format!("{:.3}", l.p95_ms),
                    format!("{:.3}", l.p99_ms),
                ]
            },
        );
        writeln!(
            out,
            "| {} | {:.1} | {} | {} | {} | {} | {} |",
            s.name,
            s.throughput(),
            values[0],
            values[1],
            values[2],
            values[3],
            s.correctness
        )
        .unwrap();
    }
    writeln!(out, "\n## Storage lifecycle\n\n| Measure | Before vacuum | After vacuum |\n| --- | ---: | ---: |\n| SQLite database bytes | {} | {} |\n| SQLite WAL bytes | {} | {} |\n", result.db_bytes_before_vacuum, result.db_bytes_after_vacuum, result.wal_bytes_before_vacuum, result.wal_bytes_after_vacuum).unwrap();
    writeln!(out, "## Interpretation and limits\n\n- End-to-end scenarios exercise SQLite WAL, the operation journal, FTS5, HNSW, scope isolation, scoring, MMR, backup/restore, and tombstone vacuuming.\n- Latency uses nearest-rank percentiles. Throughput is measured wall-clock operations divided by the measured interval. This runner is **single-threaded**; it does not claim multi-client capacity.\n- ANN and retrieval targets use deterministic random vectors and unique text tokens. Their Recall@k validates mechanics and isolation only; it is **not** semantic memory quality.\n- Before public claims, repeat each profile at least five times on a quiet pinned host, report median and spread, and run labelled LoCoMo/LongMemEval (or equivalent) with the exact embedding model and evaluator version.\n- Keep embedding-service latency separate from MemoryStore latency.\n\n## Scenario notes\n").unwrap();
    for s in &result.scenarios {
        writeln!(out, "- **{}** — {}", s.name, s.notes).unwrap();
        if let Some(l) = &s.latency {
            writeln!(
                out,
                "  Samples: {}; min/max: {:.3}/{:.3} ms.",
                l.samples, l.min_ms, l.max_ms
            )
            .unwrap();
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profiles_have_expected_production_dimensions() {
        let smoke = Config::for_profile("smoke").unwrap();
        let standard = Config::for_profile("standard").unwrap();
        assert_eq!((smoke.records, smoke.dimensions), (1_000, 384));
        assert_eq!((standard.records, standard.dimensions), (10_000, 768));
        assert!(Config::for_profile("invalid").is_err());
    }

    #[test]
    fn latency_percentiles_use_nearest_rank() {
        let samples = [1_u64, 2, 3, 4, 5].map(Duration::from_millis);
        let summary = LatencySummary::from_durations(&samples);
        assert_eq!(summary.p50_ms, 3.0);
        assert_eq!(summary.p95_ms, 5.0);
        assert_eq!(summary.p99_ms, 5.0);
    }
}
