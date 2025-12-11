import { useCallback, useState } from "react";
import { Upload, FileSpreadsheet, CheckCircle2, AlertCircle } from "lucide-react";
import { cn } from "@/lib/utils";

interface FileUploadProps {
  onFileSelect: (file: File) => void;
  isLoading?: boolean;
  isSuccess?: boolean;
  error?: string | null;
}

export function FileUpload({ onFileSelect, isLoading, isSuccess, error }: FileUploadProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [fileName, setFileName] = useState<string | null>(null);

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setIsDragging(true);
    } else if (e.type === "dragleave") {
      setIsDragging(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const files = e.dataTransfer.files;
    if (files?.[0]) {
      const file = files[0];
      if (file.name.endsWith(".xlsx") || file.name.endsWith(".xls")) {
        setFileName(file.name);
        onFileSelect(file);
      }
    }
  }, [onFileSelect]);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setFileName(file.name);
      onFileSelect(file);
    }
  }, [onFileSelect]);

  return (
    <div
      onDragEnter={handleDrag}
      onDragLeave={handleDrag}
      onDragOver={handleDrag}
      onDrop={handleDrop}
      className={cn(
        "relative group cursor-pointer rounded-2xl border-2 border-dashed p-12 transition-all duration-300",
        isDragging 
          ? "border-primary bg-primary/5 scale-[1.02]" 
          : "border-border hover:border-primary/50 hover:bg-secondary/30",
        isSuccess && "border-primary bg-primary/10",
        error && "border-destructive bg-destructive/10"
      )}
    >
      <input
        type="file"
        accept=".xlsx,.xls"
        onChange={handleFileInput}
        className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
        disabled={isLoading}
      />

      <div className="flex flex-col items-center gap-4 text-center">
        {isLoading ? (
          <>
            <div className="w-16 h-16 rounded-full bg-primary/20 flex items-center justify-center animate-pulse">
              <FileSpreadsheet className="w-8 h-8 text-primary" />
            </div>
            <div>
              <p className="text-lg font-medium text-foreground">Processing...</p>
              <p className="text-sm text-muted-foreground mt-1">Parsing Excel file</p>
            </div>
          </>
        ) : isSuccess ? (
          <>
            <div className="w-16 h-16 rounded-full bg-primary/20 flex items-center justify-center animate-slide-up">
              <CheckCircle2 className="w-8 h-8 text-primary" />
            </div>
            <div>
              <p className="text-lg font-medium text-primary">File Uploaded Successfully</p>
              <p className="text-sm text-muted-foreground mt-1 font-mono">{fileName}</p>
            </div>
          </>
        ) : error ? (
          <>
            <div className="w-16 h-16 rounded-full bg-destructive/20 flex items-center justify-center">
              <AlertCircle className="w-8 h-8 text-destructive" />
            </div>
            <div>
              <p className="text-lg font-medium text-destructive">Upload Error</p>
              <p className="text-sm text-muted-foreground mt-1">{error}</p>
            </div>
          </>
        ) : (
          <>
            <div className={cn(
              "w-16 h-16 rounded-full bg-secondary flex items-center justify-center transition-all duration-300",
              "group-hover:bg-primary/20 group-hover:scale-110"
            )}>
              <Upload className={cn(
                "w-8 h-8 text-muted-foreground transition-colors duration-300",
                "group-hover:text-primary"
              )} />
            </div>
            <div>
              <p className="text-lg font-medium text-foreground">
                Drag & drop your Excel file
              </p>
              <p className="text-sm text-muted-foreground mt-1">
                or <span className="text-primary hover:underline">browse</span> to upload
              </p>
              <p className="text-xs text-muted-foreground mt-3 font-mono">
                Accepts .xlsx, .xls files
              </p>
            </div>
          </>
        )}
      </div>

      {/* Glow effect */}
      <div className={cn(
        "absolute inset-0 rounded-2xl opacity-0 transition-opacity duration-500 pointer-events-none",
        isDragging && "opacity-100"
      )} style={{
        background: "radial-gradient(ellipse at center, hsl(190 95% 50% / 0.1) 0%, transparent 70%)"
      }} />
    </div>
  );
}
