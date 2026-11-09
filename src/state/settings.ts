/** User settings, saved in the browser (localStorage) between sessions. */

import type { Delegate } from '../core/tracker';
import type { SignLanguage, UiLanguage } from '../i18n/strings';

export interface Settings {
  uiLanguage: UiLanguage;
  signLanguage: SignLanguage;
  cameraId: string;
  resolution: 'fast' | 'hd';
  mirror: boolean;
  delegate: Delegate;
  numHands: 1 | 2;
  showSkeleton: boolean;
  showPerformance: boolean;
  theme: 'dark' | 'light';
  captionSize: 'm' | 'l' | 'xl';
  sensitivity: number;
  minConfidence: number;
  holdMs: number;
  moveThreshold: number;
  smoothing: boolean;
  eitherHand: boolean;
  /** Use the pre-trained ASL model (when the sign language is ASL). */
  pretrained: boolean;
  /** Minimum probability for the pre-trained model to write a word. */
  pretrainedConfidence: number;
  sentencePauseMs: number;
  autoSpeak: boolean;
  voiceURI: string;
  speechRate: number;
}

function defaultUiLanguage(): UiLanguage {
  const lang = navigator.language.slice(0, 2);
  return lang === 'fr' || lang === 'ar' ? lang : 'en';
}

export function defaultSettings(): Settings {
  return {
    uiLanguage: defaultUiLanguage(),
    signLanguage: 'asl',
    cameraId: '',
    resolution: 'fast',
    mirror: true,
    delegate: 'GPU',
    numHands: 2,
    showSkeleton: true,
    showPerformance: true,
    theme: 'dark',
    captionSize: 'l',
    sensitivity: 1,
    minConfidence: 0.55,
    holdMs: 400,
    moveThreshold: 1.5,
    smoothing: true,
    eitherHand: true,
    pretrained: true,
    pretrainedConfidence: 0.5,
    sentencePauseMs: 2000,
    autoSpeak: false,
    voiceURI: '',
    speechRate: 1,
  };
}

const KEY = 'signal.settings.v1';

export function loadSettings(): Settings {
  const defaults = defaultSettings();
  try {
    const stored = JSON.parse(localStorage.getItem(KEY) ?? '{}') as Partial<Settings>;
    return { ...defaults, ...stored };
  } catch {
    return defaults;
  }
}

export function saveSettings(settings: Settings): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(settings));
  } catch {
    // Storage can be unavailable (private mode). Settings then last for this session only.
  }
}
