# Figma Design Requirements: Hand Sign Recognition Web App

## 1. Overview
A web-based application that translates hand gestures (sign language) into text using a PC webcam. The app utilizes a pre-trained machine learning model to recognize 10 pre-defined words in real-time.

## 2. Design Aesthetics & Vibe
- **Style:** Modern, sleek, and highly accessible.
- **Theme:** Dark mode by default (e.g., deep charcoal or slate backgrounds) is highly recommended. It provides better contrast for the webcam video feed and reduces eye strain.
- **Color Palette:** High-contrast text with vibrant accent colors (e.g., electric blue, neon green, or warm yellow) to indicate active recognition and system status.
- **Typography:** Clean, sans-serif fonts (e.g., Inter, Roboto, or Outfit) with large, legible sizing for translated text.

## 3. Core Screens & Layout

### 3.1. Main Interface (Desktop/Web App)
- **Header:** 
  - App Logo / Title.
  - Controls: Camera toggle (on/off), Theme toggle (light/dark), Settings icon.
- **Webcam Viewport (Primary Focus):** 
  - A large central or left-aligned area displaying the live video feed.
  - Needs subtle UI overlays for: bounding boxes around detected hands, skeletal landmarks (optional), and camera loading/error states.
- **Real-Time Translation Display:** 
  - A prominent, large typography area (either overlaid at the bottom of the video feed or placed directly below it) showing the currently recognized word.
- **Vocabulary Panel (Sidebar or Bottom Grid):**
  - A dedicated section listing the 10 pre-defined words the system can recognize.
  - **States:** Highlight the active word when the corresponding gesture is recognized. Use subtle micro-animations for state changes.

### 3.2. Key UI Components & States
- **Camera Permissions:** Clean modal or inline prompt requesting webcam access.
- **Status Indicators:** 
  - 🔴 Offline / No Camera
  - 🟡 Initializing / Loading Model
  - 🟢 Ready / Detecting Hand
- **Empty State:** What the UI looks like before a hand is detected (e.g., "Show a hand gesture to start").

## 4. User Experience (UX) Flow
1. **Onboarding:** User visits the app and grants camera permissions.
2. **Idle:** The webcam feed is active, waiting for input.
3. **Action:** User performs one of the 10 hand gestures.
4. **Feedback:** The UI instantly highlights the detected word in the vocabulary panel and displays it in the main translation area with a smooth transition.

## 5. Responsive Considerations
- Although primary usage is via PC webcam (landscape orientation), the layout should be fluid.
- On smaller screens or resized windows, the Vocabulary Panel should stack below the Webcam Viewport.
