/**
 * Text-to-speech with the voices built into the operating system
 * (Web Speech API). Works offline with local voices.
 */

export function listVoices(): SpeechSynthesisVoice[] {
  if (!('speechSynthesis' in window)) return [];
  return window.speechSynthesis.getVoices();
}

/** Calls `callback` whenever the list of voices changes (it loads asynchronously). */
export function onVoicesChanged(callback: () => void): () => void {
  if (!('speechSynthesis' in window)) return () => {};
  window.speechSynthesis.addEventListener('voiceschanged', callback);
  return () => window.speechSynthesis.removeEventListener('voiceschanged', callback);
}

/** Best voice for a language: the chosen one, otherwise any voice of that language. */
export function pickVoice(lang: string, voiceURI: string): SpeechSynthesisVoice | null {
  const voices = listVoices();
  return (
    voices.find((v) => v.voiceURI === voiceURI) ??
    voices.find((v) => v.lang.toLowerCase().startsWith(lang.toLowerCase())) ??
    null
  );
}

export function speak(text: string, lang: string, voiceURI: string, rate: number): void {
  if (!('speechSynthesis' in window) || !text.trim()) return;
  const utterance = new SpeechSynthesisUtterance(text);
  const voice = pickVoice(lang, voiceURI);
  if (voice) utterance.voice = voice;
  utterance.lang = voice?.lang ?? lang;
  utterance.rate = rate;
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(utterance);
}
