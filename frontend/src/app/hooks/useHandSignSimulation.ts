import { useState, useEffect } from 'react';

export const VOCABULARY = [
  "안녕하세요", "감사합니다", "네", "아니요", "부탁합니다", 
  "죄송합니다", "도와주세요", "좋아요", "나빠요", "사랑합니다"
];

export function useHandSignSimulation(isReady: boolean, threshold: number = 80) {
  const [activeWord, setActiveWord] = useState<string | null>(null);
  const [confidence, setConfidence] = useState<number>(0);
  const [boundingBox, setBoundingBox] = useState<{ x: number, y: number, width: number, height: number } | null>(null);

  useEffect(() => {
    if (!isReady) {
      setActiveWord(null);
      setBoundingBox(null);
      return;
    }

    let interval: ReturnType<typeof setInterval>;

    const simulateDetection = () => {
      // Pick a random simulated confidence between 50 and 99
      const simulatedConfidence = Math.floor(Math.random() * 50) + 50; 
      
      // If below threshold or random 30% chance of NO hand
      if (simulatedConfidence < threshold || Math.random() < 0.3) {
        setActiveWord(null);
        setBoundingBox(null);
        return;
      }

      // Pick random word
      const word = VOCABULARY[Math.floor(Math.random() * VOCABULARY.length)];
      setActiveWord(word);
      setConfidence(simulatedConfidence); // Show actual simulated confidence

      // Random bounding box near center
      const x = 30 + Math.random() * 20; // 30-50%
      const y = 30 + Math.random() * 20; // 30-50%
      const width = 20 + Math.random() * 10; // 20-30%
      const height = 30 + Math.random() * 10; // 30-40%
      setBoundingBox({ x, y, width, height });
    };

    // Initial delay then simulate every 3 seconds
    const timeout = setTimeout(() => {
      simulateDetection();
      interval = setInterval(simulateDetection, 3000);
    }, 2000);

    return () => {
      clearTimeout(timeout);
      if (interval) clearInterval(interval);
    };
  }, [isReady]);

  return { activeWord, confidence, boundingBox };
}
