import { useState, useEffect, RefObject } from 'react';

// From the KSLR documentation
export const VOCABULARY: Record<number, string> = {
  0: "안녕하세요", 
  1: "감사합니다", 
  2: "사랑", 
  3: "학교", 
  4: "친구", 
  5: "물", 
  6: "밥", 
  7: "가다", 
  8: "오다", 
  9: "끝"
};

export function useHandSignRecognition(
  videoRef: RefObject<HTMLVideoElement | null>,
  isReady: boolean, 
  threshold: number = 80
) {
  const [activeWord, setActiveWord] = useState<string | null>(null);
  const [confidence, setConfidence] = useState<number>(0);
  const [boundingBox, setBoundingBox] = useState<{ x: number, y: number, width: number, height: number } | null>(null);

  useEffect(() => {
    if (!isReady || !videoRef.current) {
      setActiveWord(null);
      setBoundingBox(null);
      return;
    }

    const ws = new WebSocket('ws://127.0.0.1:8000/ws/default_session');
    
    ws.onopen = () => {
      console.log("WebSocket connected to ML Backend");
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        const confPct = Math.round(data.confidence * 100);
        
        if (confPct >= threshold) {
          const word = VOCABULARY[data.label_id];
          setActiveWord(word || null);
          setConfidence(confPct);
        } else {
          setActiveWord(null);
        }
      } catch (err) {
        console.error("Error parsing websocket message", err);
      }
    };

    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    let interval: ReturnType<typeof setInterval>;

    const sendFrame = () => {
      if (ws.readyState !== WebSocket.OPEN) return;
      const video = videoRef.current;
      if (!video || video.videoWidth === 0) return;

      // Maintain a reasonable resolution for inference
      const targetWidth = 640;
      const targetHeight = (video.videoHeight / video.videoWidth) * targetWidth;

      canvas.width = targetWidth;
      canvas.height = targetHeight;
      
      if (ctx) {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        canvas.toBlob((blob) => {
          if (blob && ws.readyState === WebSocket.OPEN) {
            ws.send(blob);
          }
        }, 'image/jpeg', 0.8);
      }
    };

    // Send frames at roughly 15-20 FPS
    interval = setInterval(sendFrame, 1000 / 15);

    return () => {
      clearInterval(interval);
      ws.close();
    };
  }, [isReady, threshold, videoRef]);

  return { activeWord, confidence, boundingBox };
}
