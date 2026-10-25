import * as crypto from 'node:crypto';

/**
 * Common interface for embedding adapters.
 */
export interface Embedder {
  readonly name: string;
  readonly dimensions: number;
  embed(text: string): Promise<Float32Array>;
  embedBatch(texts: string[]): Promise<Float32Array[]>;
}

/**
 * Deterministic, zero-dependency embedding generator for testing and offline development.
 * Generates reproducible unit-norm vector projections from text strings using SHA-256.
 */
export class DeterministicEmbedder implements Embedder {
  public readonly name: string;
  public readonly dimensions: number;

  constructor(name = 'test-model', dimensions = 1536) {
    if (dimensions <= 0) {
      throw new Error(`dimensions must be positive, got ${dimensions}`);
    }
    this.name = name;
    this.dimensions = dimensions;
  }

  async embed(text: string): Promise<Float32Array> {
    const dims = this.dimensions;
    const vec = new Float32Array(dims);

    const baseHash = crypto.createHash('sha256').update(text, 'utf8').digest();

    for (let i = 0; i < dims; i++) {
      const idxBuf = Buffer.alloc(4);
      idxBuf.writeInt32LE(i, 0);

      const compHash = crypto
        .createHash('sha256')
        .update(baseHash)
        .update(idxBuf)
        .digest();

      const val = compHash.readInt32LE(0);
      vec[i] = val / 2147483648.0;
    }

    // Normalize to unit vector
    let sumSq = 0;
    for (let i = 0; i < dims; i++) {
      sumSq += vec[i] * vec[i];
    }
    const norm = Math.sqrt(sumSq);
    if (norm > 0) {
      for (let i = 0; i < dims; i++) {
        vec[i] /= norm;
      }
    }

    return vec;
  }

  async embedBatch(texts: string[]): Promise<Float32Array[]> {
    return Promise.all(texts.map((t) => this.embed(t)));
  }
}

/**
 * OpenAI embedding adapter using standard native fetch.
 */
export class OpenAIEmbedder implements Embedder {
  public static readonly KNOWN_DIMENSIONS: Record<string, number> = {
    'text-embedding-3-small': 1536,
    'text-embedding-3-large': 3072,
    'text-embedding-ada-002': 1536,
  };

  public readonly name: string;
  private _dimensions: number;
  private _apiKey: string;
  private _baseUrl: string;

  constructor(options?: {
    model?: string;
    dimensions?: number;
    apiKey?: string;
    baseUrl?: string;
  }) {
    this.name = options?.model ?? 'text-embedding-3-small';
    this._dimensions = options?.dimensions ?? OpenAIEmbedder.KNOWN_DIMENSIONS[this.name] ?? 1536;
    this._apiKey = options?.apiKey ?? (typeof process !== 'undefined' ? process.env.OPENAI_API_KEY || '' : '');
    this._baseUrl = options?.baseUrl ?? 'https://api.openai.com/v1';

    if (!this._apiKey) {
      // Allowed to be initialized without API key if not yet making calls
    }
  }

  get dimensions(): number {
    return this._dimensions;
  }

  async embed(text: string): Promise<Float32Array> {
    const results = await this.embedBatch([text]);
    return results[0];
  }

  async embedBatch(texts: string[]): Promise<Float32Array[]> {
    if (!this._apiKey) {
      throw new Error('OpenAIEmbedder: OPENAI_API_KEY is required to generate embeddings.');
    }

    const payload: Record<string, unknown> = {
      model: this.name,
      input: texts,
    };
    if (this.name.startsWith('text-embedding-3')) {
      payload.dimensions = this._dimensions;
    }

    const res = await fetch(`${this._baseUrl}/embeddings`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${this._apiKey}`,
      },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const errText = await res.text();
      throw new Error(`OpenAI API error [${res.status}]: ${errText}`);
    }

    const data = (await res.json()) as { data: Array<{ embedding: number[] }> };
    return data.data.map((item) => new Float32Array(item.embedding));
  }
}

/**
 * Local Ollama embedding adapter using native fetch.
 */
export class OllamaEmbedder implements Embedder {
  public readonly name: string;
  private _dimensions: number;
  private _baseUrl: string;

  constructor(options?: {
    model?: string;
    dimensions?: number;
    baseUrl?: string;
  }) {
    this.name = options?.model ?? 'nomic-embed-text';
    this._dimensions = options?.dimensions ?? 768;
    this._baseUrl = options?.baseUrl ?? 'http://localhost:11434';
  }

  get dimensions(): number {
    return this._dimensions;
  }

  async embed(text: string): Promise<Float32Array> {
    const res = await fetch(`${this._baseUrl}/api/embeddings`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: this.name, prompt: text }),
    });

    if (!res.ok) {
      const errText = await res.text();
      throw new Error(`Ollama API error [${res.status}]: ${errText}`);
    }

    const data = (await res.json()) as { embedding: number[] };
    const arr = new Float32Array(data.embedding);
    if (!this._dimensions || this._dimensions !== arr.length) {
      this._dimensions = arr.length;
    }
    return arr;
  }

  async embedBatch(texts: string[]): Promise<Float32Array[]> {
    return Promise.all(texts.map((t) => this.embed(t)));
  }
}
