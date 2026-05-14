const KUROMOJI_DICT_PATH = "https://unpkg.com/kuromoji@0.1.2/dict/";

let converterPromise = null;

self.addEventListener("message", async ({ data }) => {
  const id = data?.id;
  const text = data?.text;
  if (typeof id !== "number" || typeof text !== "string") return;

  try {
    const converter = await getConverter();
    const result = await converter.convert(text, {
      to: "hiragana",
      mode: "furigana_map",
      includeKatakana: false,
    });
    self.postMessage({ id, result });
  } catch (err) {
    self.postMessage({ id, error: err?.message ?? String(err) });
  }
});

function getConverter() {
  converterPromise ??= (async () => {
    importScripts(
      "https://unpkg.com/kuroshiro-enhance@2.0.0/dist/kuroshiro.min.js",
      "https://unpkg.com/kuroshiro-analyzer-kuromoji@1.1.0/dist/kuroshiro-analyzer-kuromoji.min.js",
    );

    const KuroshiroClass = self.Kuroshiro?.default || self.Kuroshiro;
    const KuromojiAnalyzerClass = self.KuromojiAnalyzer?.default || self.KuromojiAnalyzer;
    if (!KuroshiroClass || !KuromojiAnalyzerClass) {
      throw new Error("Kuroshiro scripts did not load");
    }

    const converter = new KuroshiroClass();
    await converter.init(new KuromojiAnalyzerClass({ dictPath: KUROMOJI_DICT_PATH }));
    return converter;
  })();

  return converterPromise;
}
