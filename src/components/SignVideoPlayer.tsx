/**
 * Plays a list of sign clips one after the other, at the chosen speed,
 * optionally mirrored (so the signer's right hand is on the same side as
 * yours, like a mirror) and looped.
 *
 * Two <video> elements take turns: while one plays, the other already loads
 * the next clip, so a sentence plays without a pause between signs.
 */

import { useEffect, useRef, useState } from 'react';

export interface VideoClip {
  src: string;
  /** Fingerspelled letters play faster: a signer spells quicker than they sign. */
  letter?: boolean;
}

const LETTER_SPEEDUP = 1.6;

export function SignVideoPlayer({
  clips,
  speed = 1,
  mirror = false,
  loop = false,
  playing = true,
  replayKey = 0,
  onClipChange,
  onEnd,
  label,
  failedLabel,
}: {
  clips: VideoClip[];
  speed?: number;
  mirror?: boolean;
  loop?: boolean;
  playing?: boolean;
  /** Changing this number restarts playback from the first clip. */
  replayKey?: number;
  onClipChange?: (index: number) => void;
  onEnd?: () => void;
  label?: string;
  /** Message shown when a clip cannot be played. */
  failedLabel?: string;
}) {
  const refs = [useRef<HTMLVideoElement>(null), useRef<HTMLVideoElement>(null)];
  // Which of the two elements is on screen, and which clip it shows.
  const [front, setFront] = useState(0);
  const [index, setIndex] = useState(0);
  const [failed, setFailed] = useState(false);
  const settings = useRef({ speed, loop, playing, onClipChange, onEnd });
  settings.current = { speed, loop, playing, onClipChange, onEnd };

  const rate = (i: number) => speed * (clips[i]?.letter ? LETTER_SPEEDUP : 1);

  // Puts clip i into element e (without playing it).
  const load = (e: number, i: number) => {
    const v = refs[e]!.current;
    const clip = clips[i];
    if (!v || !clip) return;
    if (v.getAttribute('src') !== clip.src) v.src = clip.src;
    else v.currentTime = 0;
  };

  // Start again from the first clip when the clips or the replay key change.
  useEffect(() => {
    setFailed(false);
    setFront(0);
    setIndex(0);
    load(0, 0);
    if (clips.length > 1) load(1, 1);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [clips, replayKey]);

  // Tell the parent which clip is showing.
  useEffect(() => {
    if (clips.length) settings.current.onClipChange?.(index);
  }, [index, clips, replayKey]);

  // Play or pause the element on screen, at the right speed.
  useEffect(() => {
    const v = refs[front]!.current;
    if (!v || !clips.length) return;
    v.playbackRate = rate(index);
    if (playing) void v.play().catch(() => undefined);
    else v.pause();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [front, index, playing, speed, clips, replayKey]);

  const next = () => {
    const s = settings.current;
    const last = index + 1 >= clips.length;
    if (last && !s.loop) {
      s.onEnd?.();
      return;
    }
    const target = last ? 0 : index + 1;
    setFailed(false);
    if (clips.length === 1) {
      // A single looping clip: just start it again.
      const v = refs[front]!.current;
      if (v) {
        v.currentTime = 0;
        void v.play().catch(() => undefined);
      }
      return;
    }
    // The other element already holds the next clip (except when looping back).
    const back = 1 - front;
    if (last) load(back, 0);
    setFront(back);
    setIndex(target);
    // Load the clip after that into the element just freed.
    const after = target + 1 < clips.length ? target + 1 : s.loop ? 0 : -1;
    if (after >= 0) load(front, after);
  };

  return (
    <div className="player player-video-wrap">
      {[0, 1].map((e) => (
        <video
          key={e}
          ref={refs[e]}
          className={`player-video ${mirror ? 'is-mirrored' : ''}`}
          hidden={e !== front}
          muted
          playsInline
          preload="auto"
          aria-label={e === front ? label : undefined}
          onEnded={e === front ? next : undefined}
          onError={
            e === front
              ? () => {
                  // Say so, and go on with the next sign rather than stopping.
                  setFailed(true);
                  setTimeout(next, 600);
                }
              : undefined
          }
        />
      ))}
      {failed && <p className="muted player-error">{failedLabel}</p>}
    </div>
  );
}
