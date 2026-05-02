function bytesToUnicode() {
  const bs = [];
  for (let i = 33; i <= 126; ++i) bs.push(i);
  for (let i = 161; i <= 172; ++i) bs.push(i);
  for (let i = 174; i <= 255; ++i) bs.push(i);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; ++b) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n);
      n += 1;
    }
  }
  const byteEncoder = new Map();
  const byteDecoder = new Map();
  for (let i = 0; i < bs.length; ++i) {
    const ch = String.fromCodePoint(cs[i]);
    byteEncoder.set(bs[i], ch);
    byteDecoder.set(ch, bs[i]);
  }
  return { byteEncoder, byteDecoder };
}

function getPairs(word) {
  const pairs = new Set();
  for (let i = 0; i + 1 < word.length; ++i) {
    pairs.add(`${word[i]}\u0000${word[i + 1]}`);
  }
  return pairs;
}

export class QwenTokenizer {
  static async from_pretrained(baseUrl) {
    const root = baseUrl.endsWith("/") ? baseUrl : `${baseUrl}/`;
    const [vocab, mergesText, tokenizerConfig] = await Promise.all([
      fetch(`${root}vocab.json`).then((r) => r.json()),
      fetch(`${root}merges.txt`).then((r) => r.text()),
      fetch(`${root}tokenizer_config.json`).then((r) => r.json()),
    ]);
    return new QwenTokenizer(vocab, mergesText, tokenizerConfig);
  }

  constructor(vocab, mergesText, tokenizerConfig) {
    this.vocab = vocab;
    this.idToToken = new Map(Object.entries(vocab).map(([token, id]) => [id, token]));
    this.cache = new Map();
    const { byteEncoder, byteDecoder } = bytesToUnicode();
    this.byteEncoder = byteEncoder;
    this.byteDecoder = byteDecoder;
    this.encoder = new TextEncoder();
    this.decoder = new TextDecoder("utf-8", { fatal: false });
    this.specialTokens = new Map();
    for (const [id, entry] of Object.entries(tokenizerConfig.added_tokens_decoder || {})) {
      this.specialTokens.set(entry.content, Number.parseInt(id, 10));
    }
    this.idToSpecial = new Map([...this.specialTokens.entries()].map(([token, id]) => [id, token]));
    this.bpeRanks = new Map();
    let rank = 0;
    for (const line of mergesText.split(/\r?\n/)) {
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith("#")) continue;
      const parts = trimmed.split(/\s+/);
      if (parts.length === 2) {
        this.bpeRanks.set(`${parts[0]}\u0000${parts[1]}`, rank++);
      }
    }
    this.pattern = /'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;
  }

  applyChatTemplate(userText) {
    return `<|im_start|>user\n${userText}<|im_end|>\n<|im_start|>assistant\n`;
  }

  encode(text) {
    const ids = [];
    let i = 0;
    const specials = [...this.specialTokens.keys()].sort((a, b) => b.length - a.length);
    while (i < text.length) {
      let matched = false;
      for (const special of specials) {
        if (text.startsWith(special, i)) {
          ids.push(this.specialTokens.get(special));
          i += special.length;
          matched = true;
          break;
        }
      }
      if (matched) continue;
      let nextSpecial = text.length;
      for (const special of specials) {
        const pos = text.indexOf(special, i);
        if (pos >= 0) nextSpecial = Math.min(nextSpecial, pos);
      }
      const chunk = text.slice(i, nextSpecial);
      for (const match of chunk.matchAll(this.pattern)) {
        const piece = match[0];
        const bytes = this.encoder.encode(piece);
        let token = "";
        for (const b of bytes) token += this.byteEncoder.get(b);
        for (const bpeToken of this.bpe(token)) {
          const id = this.vocab[bpeToken];
          if (id === undefined) throw new Error(`Tokenizer missing token: ${bpeToken}`);
          ids.push(id);
        }
      }
      i = nextSpecial;
    }
    return ids;
  }

  bpe(token) {
    const cached = this.cache.get(token);
    if (cached) return cached;
    let word = [...token];
    if (word.length <= 1) {
      this.cache.set(token, word);
      return word;
    }
    while (true) {
      let bestPair = null;
      let bestRank = Number.POSITIVE_INFINITY;
      for (const pair of getPairs(word)) {
        const rank = this.bpeRanks.get(pair);
        if (rank !== undefined && rank < bestRank) {
          bestRank = rank;
          bestPair = pair;
        }
      }
      if (bestPair === null) break;
      const [first, second] = bestPair.split("\u0000");
      const next = [];
      for (let i = 0; i < word.length; ++i) {
        if (i + 1 < word.length && word[i] === first && word[i + 1] === second) {
          next.push(first + second);
          i += 1;
        } else {
          next.push(word[i]);
        }
      }
      word = next;
      if (word.length === 1) break;
    }
    this.cache.set(token, word);
    return word;
  }

  decode(ids, { skipSpecialTokens = false } = {}) {
    const bytes = [];
    let text = "";
    const flush = () => {
      if (bytes.length) {
        text += this.decoder.decode(new Uint8Array(bytes));
        bytes.length = 0;
      }
    };
    for (const id of ids) {
      if (this.idToSpecial.has(id)) {
        flush();
        if (!skipSpecialTokens) text += this.idToSpecial.get(id);
        continue;
      }
      const token = this.idToToken.get(id);
      if (token === undefined) {
        flush();
        text += `<${id}>`;
        continue;
      }
      for (const ch of [...token]) {
        const b = this.byteDecoder.get(ch);
        if (b !== undefined) bytes.push(b);
      }
    }
    flush();
    return text;
  }
}
