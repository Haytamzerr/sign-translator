/** The camera view: video, hand skeleton overlay, and status messages on top. */

import { useEffect, useRef } from 'react';
import type { Translate } from '../i18n/strings';
import type { EngineSnapshot, SignEngine } from '../state/engine';
import type { Settings } from '../state/settings';
import { IconCamera } from './icons';

export interface TeachOverlay {
  phase: 'countdown' | 'recording';
  count: number;
  progress: number;
  take: number;
  total: number;
}

const ERROR_KEYS = { denied: 'cameraDenied', missing: 'cameraMissing', busy: 'cameraBusy', model: 'modelFailed' } as const;

export function Stage({
  engine,
  snap,
  settings,
  teach,
  t,
}: {
  engine: SignEngine;
  snap: EngineSnapshot;
  settings: Settings;
  teach: TeachOverlay | null;
  t: Translate;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (videoRef.current && canvasRef.current) engine.attach(videoRef.current, canvasRef.current);
  }, [engine]);

  const stateLabel = { idle: t('stateIdle'), hold: t('stateHold'), moving: t('stateMoving') }[snap.state];

  return (
    <div className={`stage ${settings.mirror ? 'is-mirrored' : ''}`} data-status={snap.status}>
      <div className="stage-media">
        <video ref={videoRef} playsInline muted />
        <canvas ref={canvasRef} />
      </div>

      {snap.status === 'running' && (
        <div className="stage-chips">
          <span className={`chip chip-state is-${snap.state}`}>
            <span className="dot" aria-hidden="true" />
            {stateLabel}
          </span>
          {settings.showPerformance && (
            <span className="chip chip-perf" title="fps / inference time">
              {Math.round(snap.fps)} {t('fps')} · {snap.inferenceMs.toFixed(1)} {t('ms')}
            </span>
          )}
        </div>
      )}

      {snap.status === 'off' && (
        <div className="stage-message">
          <h2>{t('cameraOffTitle')}</h2>
          <p>{t('cameraOffBody')}</p>
          <button className="btn btn-primary" onClick={() => void engine.start()}>
            <IconCamera /> {t('cameraStart')}
          </button>
        </div>
      )}

      {snap.status === 'loading' && (
        <div className="stage-message">
          <div className="spinner" aria-hidden="true" />
          <p>{t('loadingModel')}</p>
        </div>
      )}

      {snap.status === 'error' && snap.error && (
        <div className="stage-message" role="alert">
          <p>{t(ERROR_KEYS[snap.error])}</p>
          <button className="btn" onClick={() => void engine.start()}>
            {t('tryAgain')}
          </button>
        </div>
      )}

      {teach && (
        <div className="teach-overlay" aria-live="assertive">
          {teach.phase === 'countdown' ? (
            <>
              <span className="teach-count">{teach.count}</span>
              <span className="teach-caption">{t('teachGetReady')}</span>
            </>
          ) : (
            <>
              <span className="rec-badge">
                <span className="dot" aria-hidden="true" />
                {t('teachRecording', { n: teach.take, total: teach.total })}
              </span>
              <span className="rec-progress" style={{ transform: `scaleX(${teach.progress})` }} />
            </>
          )}
        </div>
      )}
    </div>
  );
}
