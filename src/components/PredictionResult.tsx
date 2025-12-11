import { cn } from "@/lib/utils";
import { Zap, Shield } from "lucide-react";

interface PredictionResultProps {
  prediction: "AIR" | "METAL" | null;
  confidence: number;
  isLoading?: boolean;
}

export function PredictionResult({ prediction, confidence, isLoading }: PredictionResultProps) {
  if (isLoading) {
    return (
      <div className="glass rounded-2xl p-8 text-center">
        <div className="flex items-center justify-center gap-3">
          <div className="w-3 h-3 rounded-full bg-primary animate-pulse" />
          <div className="w-3 h-3 rounded-full bg-primary animate-pulse" style={{ animationDelay: "0.2s" }} />
          <div className="w-3 h-3 rounded-full bg-primary animate-pulse" style={{ animationDelay: "0.4s" }} />
        </div>
        <p className="text-muted-foreground mt-4">Running classification model...</p>
      </div>
    );
  }

  if (!prediction) {
    return (
      <div className="glass rounded-2xl p-8 text-center border-dashed">
        <p className="text-muted-foreground">
          Upload an Excel file to see the classification result
        </p>
      </div>
    );
  }

  const isAir = prediction === "AIR";
  const confidencePercent = Math.round(confidence * 100);

  return (
    <div className={cn(
      "rounded-2xl p-8 text-center animate-slide-up transition-all duration-500",
      isAir ? "glass glow-air" : "glass glow-metal"
    )}>
      {/* Icon */}
      <div className={cn(
        "w-20 h-20 rounded-full mx-auto mb-6 flex items-center justify-center",
        isAir ? "bg-air/20" : "bg-metal/20"
      )}>
        {isAir ? (
          <Zap className="w-10 h-10 text-air animate-pulse-glow" />
        ) : (
          <Shield className="w-10 h-10 text-metal animate-pulse-glow" />
        )}
      </div>

      {/* Prediction Label */}
      <h2 className={cn(
        "text-5xl font-bold mb-2 tracking-tight",
        isAir ? "text-gradient-air" : "text-gradient-metal"
      )}>
        {prediction}
      </h2>
      
      <p className="text-muted-foreground text-sm mb-6">
        Predicted Classification
      </p>

      {/* Confidence Bar */}
      <div className="max-w-xs mx-auto">
        <div className="flex justify-between items-center mb-2">
          <span className="text-sm text-muted-foreground">Confidence</span>
          <span className={cn(
            "text-lg font-bold font-mono",
            isAir ? "text-air" : "text-metal"
          )}>
            {confidencePercent}%
          </span>
        </div>
        <div className="h-3 bg-secondary rounded-full overflow-hidden">
          <div
            className={cn(
              "h-full rounded-full transition-all duration-1000 ease-out",
              isAir 
                ? "bg-gradient-to-r from-air/80 to-air" 
                : "bg-gradient-to-r from-metal/80 to-metal"
            )}
            style={{ width: `${confidencePercent}%` }}
          />
        </div>
      </div>

      {/* Confidence Level Indicator */}
      <div className="mt-4">
        <span className={cn(
          "inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium",
          confidence >= 0.9 
            ? "bg-green-500/20 text-green-400" 
            : confidence >= 0.7 
              ? "bg-yellow-500/20 text-yellow-400"
              : "bg-orange-500/20 text-orange-400"
        )}>
          {confidence >= 0.9 ? "High Confidence" : confidence >= 0.7 ? "Medium Confidence" : "Low Confidence"}
        </span>
      </div>
    </div>
  );
}
