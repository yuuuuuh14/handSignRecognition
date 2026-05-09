import { useState, useEffect, RefObject } from 'react';

export type CameraStatus = 'offline' | 'initializing' | 'ready' | 'error';

export function useWebcam(videoRef: RefObject<HTMLVideoElement | null>, enabled: boolean) {
  const [status, setStatus] = useState<CameraStatus>('offline');
  const [errorMsg, setErrorMsg] = useState<string>('');

  useEffect(() => {
    let stream: MediaStream | null = null;

    async function setupCamera() {
      if (!enabled) {
        if (stream) {
          stream.getTracks().forEach(track => track.stop());
        }
        if (videoRef.current) {
          videoRef.current.srcObject = null;
        }
        setStatus('offline');
        return;
      }

      setStatus('initializing');
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: false,
        });
        
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.onloadedmetadata = () => {
            videoRef.current?.play();
            setStatus('ready');
          };
        }
      } catch (err: any) {
        console.error("Error accessing webcam", err);
        setStatus('error');
        setErrorMsg(err.message || "Failed to access webcam. Please check permissions.");
      }
    }

    setupCamera();

    return () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, [enabled, videoRef]);

  return { status, errorMsg };
}
