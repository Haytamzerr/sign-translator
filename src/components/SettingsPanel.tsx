/** All settings, grouped into sections. Every change applies immediately. */

import { useEffect, useState } from 'react';
import { listVoices, onVoicesChanged } from '../core/speech';
import { SIGN_LANGUAGES, speechLang, type Translate, type UiLanguage } from '../i18n/strings';
import { defaultSettings, type Settings } from '../state/settings';
import { Field, Segmented, Select, Slider, Toggle } from './controls';

export function SettingsPanel({
  settings,
  update,
  cameraOn,
  t,
}: {
  settings: Settings;
  update: (patch: Partial<Settings>) => void;
  cameraOn: boolean;
  t: Translate;
}) {
  const [cameras, setCameras] = useState<MediaDeviceInfo[]>([]);
  const [voices, setVoices] = useState<SpeechSynthesisVoice[]>(listVoices());

  // Camera names are only visible after the camera has been allowed once.
  useEffect(() => {
    navigator.mediaDevices
      ?.enumerateDevices()
      .then((devices) => setCameras(devices.filter((d) => d.kind === 'videoinput')))
      .catch(() => setCameras([]));
  }, [cameraOn]);

  useEffect(() => onVoicesChanged(() => setVoices(listVoices())), []);

  const lang = speechLang[settings.uiLanguage];
  const langVoices = voices.filter((v) => v.lang.toLowerCase().startsWith(lang));

  return (
    <div className="panel-body settings">
      <section className="block">
        <h3 className="block-title">{t('settingsLanguage')}</h3>
        <Field label={t('settingInterfaceLanguage')}>
          <Segmented<UiLanguage>
            label={t('settingInterfaceLanguage')}
            value={settings.uiLanguage}
            options={[
              { value: 'en', label: 'English' },
              { value: 'fr', label: 'Français' },
              { value: 'ar', label: 'العربية' },
            ]}
            onChange={(uiLanguage) => update({ uiLanguage, voiceURI: '' })}
          />
        </Field>
        <Field label={t('settingSignLanguage')} help={t('settingSignLanguageHelp')}>
          <Select
            label={t('settingSignLanguage')}
            value={settings.signLanguage}
            options={SIGN_LANGUAGES.map((value) => ({ value, label: t(`sl_${value}`) }))}
            onChange={(signLanguage) => update({ signLanguage })}
          />
        </Field>
      </section>

      <section className="block">
        <h3 className="block-title">{t('settingsCamera')}</h3>
        <Field label={t('settingCameraDevice')}>
          <Select
            label={t('settingCameraDevice')}
            value={settings.cameraId}
            options={[
              { value: '', label: t('settingDefaultCamera') },
              ...cameras.filter((c) => c.deviceId).map((c, i) => ({ value: c.deviceId, label: c.label || `Camera ${i + 1}` })),
            ]}
            onChange={(cameraId) => update({ cameraId })}
          />
        </Field>
        <Field label={t('settingResolution')}>
          <Segmented
            label={t('settingResolution')}
            value={settings.resolution}
            options={[
              { value: 'fast', label: t('resolutionFast') },
              { value: 'hd', label: t('resolutionHd') },
            ]}
            onChange={(resolution) => update({ resolution })}
          />
        </Field>
        <Field label={t('settingProcessor')}>
          <Segmented
            label={t('settingProcessor')}
            value={settings.delegate}
            options={[
              { value: 'GPU', label: t('processorGpu') },
              { value: 'CPU', label: t('processorCpu') },
            ]}
            onChange={(delegate) => update({ delegate })}
          />
        </Field>
        <Field label={t('settingHands')}>
          <Segmented
            label={t('settingHands')}
            value={settings.numHands}
            options={[
              { value: 1 as const, label: t('oneHand') },
              { value: 2 as const, label: t('twoHands') },
            ]}
            onChange={(numHands) => update({ numHands })}
          />
        </Field>
        <Toggle label={t('settingMirror')} checked={settings.mirror} onChange={(mirror) => update({ mirror })} />
      </section>

      <section className="block">
        <h3 className="block-title">{t('settingsRecognition')}</h3>
        <Field label={t('settingSensitivity')} help={t('settingSensitivityHelp')}>
          <Slider
            label={t('settingSensitivity')}
            value={settings.sensitivity}
            min={0.6}
            max={1.6}
            step={0.05}
            format={(v) => `${Math.round(v * 100)}%`}
            onChange={(sensitivity) => update({ sensitivity })}
          />
        </Field>
        <Field label={t('settingConfidence')}>
          <Slider
            label={t('settingConfidence')}
            value={settings.minConfidence}
            min={0.3}
            max={0.9}
            step={0.05}
            format={(v) => `${Math.round(v * 100)}%`}
            onChange={(minConfidence) => update({ minConfidence })}
          />
        </Field>
        <Field label={t('settingHold')}>
          <Slider
            label={t('settingHold')}
            value={settings.holdMs}
            min={150}
            max={1200}
            step={50}
            format={(v) => `${(v / 1000).toFixed(2)} s`}
            onChange={(holdMs) => update({ holdMs })}
          />
        </Field>
        <Field label={t('settingMotion')} help={t('settingMotionHelp')}>
          <Slider
            label={t('settingMotion')}
            value={settings.moveThreshold}
            min={0.6}
            max={4}
            step={0.1}
            format={(v) => v.toFixed(1)}
            onChange={(moveThreshold) => update({ moveThreshold })}
          />
        </Field>
        {settings.signLanguage === 'asl' && (
          <>
            <div className="field">
              <Toggle label={t('settingPretrained')} checked={settings.pretrained} onChange={(pretrained) => update({ pretrained })} />
              <p className="field-help">{t('settingPretrainedHelp')}</p>
            </div>
            <Field label={t('settingPretrainedConfidence')}>
              <Slider
                label={t('settingPretrainedConfidence')}
                value={settings.pretrainedConfidence}
                min={0.2}
                max={0.9}
                step={0.05}
                format={(v) => `${Math.round(v * 100)}%`}
                onChange={(pretrainedConfidence) => update({ pretrainedConfidence })}
              />
            </Field>
          </>
        )}
        <Toggle label={t('settingSmoothing')} checked={settings.smoothing} onChange={(smoothing) => update({ smoothing })} />
        <Toggle label={t('settingEitherHand')} checked={settings.eitherHand} onChange={(eitherHand) => update({ eitherHand })} />
      </section>

      <section className="block">
        <h3 className="block-title">{t('settingsCaptions')}</h3>
        <Field label={t('settingPause')}>
          <Slider
            label={t('settingPause')}
            value={settings.sentencePauseMs}
            min={800}
            max={6000}
            step={200}
            format={(v) => `${(v / 1000).toFixed(1)} s`}
            onChange={(sentencePauseMs) => update({ sentencePauseMs })}
          />
        </Field>
        <Toggle label={t('settingAutoSpeak')} checked={settings.autoSpeak} onChange={(autoSpeak) => update({ autoSpeak })} />
        <Field label={t('settingVoice')} help={langVoices.length === 0 ? t('settingNoVoice') : undefined}>
          <Select
            label={t('settingVoice')}
            value={settings.voiceURI}
            options={[{ value: '', label: t('settingAutoVoice') }, ...langVoices.map((v) => ({ value: v.voiceURI, label: v.name }))]}
            onChange={(voiceURI) => update({ voiceURI })}
          />
        </Field>
        <Field label={t('settingRate')}>
          <Slider
            label={t('settingRate')}
            value={settings.speechRate}
            min={0.6}
            max={1.6}
            step={0.1}
            format={(v) => `${v.toFixed(1)}×`}
            onChange={(speechRate) => update({ speechRate })}
          />
        </Field>
      </section>

      <section className="block">
        <h3 className="block-title">{t('settingsDisplay')}</h3>
        <Field label={t('settingTheme')}>
          <Segmented
            label={t('settingTheme')}
            value={settings.theme}
            options={[
              { value: 'dark', label: t('themeDark') },
              { value: 'light', label: t('themeLight') },
            ]}
            onChange={(theme) => update({ theme })}
          />
        </Field>
        <Field label={t('settingCaptionSize')}>
          <Segmented
            label={t('settingCaptionSize')}
            value={settings.captionSize}
            options={[
              { value: 'm', label: t('sizeM') },
              { value: 'l', label: t('sizeL') },
              { value: 'xl', label: t('sizeXL') },
            ]}
            onChange={(captionSize) => update({ captionSize })}
          />
        </Field>
        <Toggle label={t('settingSkeleton')} checked={settings.showSkeleton} onChange={(showSkeleton) => update({ showSkeleton })} />
        <Toggle
          label={t('settingPerformance')}
          checked={settings.showPerformance}
          onChange={(showPerformance) => update({ showPerformance })}
        />
      </section>

      <button
        className="btn"
        onClick={() => {
          const d = defaultSettings();
          update({ ...d, uiLanguage: settings.uiLanguage, signLanguage: settings.signLanguage, cameraId: settings.cameraId });
        }}
      >
        {t('settingsReset')}
      </button>
    </div>
  );
}
