import { Settings, Moon, Sun, Camera, CameraOff, Sparkles } from "lucide-react";

interface HeaderProps {
  cameraEnabled: boolean;
  onToggleCamera: () => void;
  onOpenSettings: () => void;
}

export function Header({ cameraEnabled, onToggleCamera, onOpenSettings }: HeaderProps) {
  return (
    <header className="flex items-center justify-between py-4 px-6 bg-white dark:bg-slate-900 border-b border-slate-200 dark:border-slate-800 transition-colors">
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-xl bg-blue-500/10 dark:bg-blue-500/20 flex items-center justify-center border border-blue-500/30 dark:border-blue-500/50">
          <Sparkles className="w-5 h-5 text-blue-600 dark:text-blue-400" />
        </div>
        <div>
          <h1 className="text-xl font-bold text-slate-900 dark:text-white tracking-tight font-outfit">SignTranslate</h1>
          <p className="text-xs text-slate-500 dark:text-slate-400 font-medium">실시간 KSL(한국 수어) 인식</p>
        </div>
      </div>

      <div className="flex items-center gap-4">
        <button 
          onClick={onToggleCamera}
          className={`flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium transition-colors ${
            cameraEnabled 
              ? "bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-200 hover:bg-slate-200 dark:hover:bg-slate-700" 
              : "bg-blue-600 text-white hover:bg-blue-500 shadow-[0_0_15px_rgba(37,99,235,0.4)]"
          }`}
        >
          {cameraEnabled ? <CameraOff className="w-4 h-4" /> : <Camera className="w-4 h-4" />}
          {cameraEnabled ? "카메라 끄기" : "카메라 켜기"}
        </button>

        <div className="h-6 w-px bg-slate-200 dark:bg-slate-800 mx-1"></div>

        <button 
          onClick={onOpenSettings}
          className="p-2 text-slate-500 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white hover:bg-slate-100 dark:hover:bg-slate-800 rounded-full transition-colors"
        >
          <Settings className="w-5 h-5" />
        </button>
      </div>
    </header>
  );
}
