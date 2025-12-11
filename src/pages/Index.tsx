import { useState, useCallback } from "react";
import { FileUpload } from "@/components/FileUpload";
import { DataTable } from "@/components/DataTable";
import { SignalChart } from "@/components/SignalChart";
import { PredictionResult } from "@/components/PredictionResult";
import { parseExcelFile, validateDataShape, ParsedExcelData } from "@/lib/excelParser";
import { useToast } from "@/hooks/use-toast";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Activity, Table2, Info, Server } from "lucide-react";

const SIGNAL_COLORS = {
  "Current (kA)": "#22d3ee",      // cyan
  "Voltage (V)": "#a78bfa",        // purple  
  "Resistance (Ohm)": "#34d399",   // green
  "Force (kg)": "#fb923c",         // orange
};

const Index = () => {
  const { toast } = useToast();
  const [isLoading, setIsLoading] = useState(false);
  const [isPredicting, setIsPredicting] = useState(false);
  const [isSuccess, setIsSuccess] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [parsedData, setParsedData] = useState<ParsedExcelData | null>(null);
  const [prediction, setPrediction] = useState<"AIR" | "METAL" | null>(null);
  const [confidence, setConfidence] = useState(0);

  const handleFileSelect = useCallback(async (file: File) => {
    setIsLoading(true);
    setError(null);
    setIsSuccess(false);
    setPrediction(null);
    setConfidence(0);

    try {
      const data = await parseExcelFile(file);
      
      // Validate data shape
      const validation = validateDataShape(data.signalData);
      if (!validation.valid) {
        throw new Error(validation.message);
      }

      setParsedData(data);
      setIsSuccess(true);
      
      toast({
        title: "File parsed successfully",
        description: validation.message,
      });

      // Simulate prediction (replace with actual backend call)
      simulatePrediction();
      
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to parse file";
      setError(message);
      toast({
        title: "Error parsing file",
        description: message,
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  }, [toast]);

  const simulatePrediction = () => {
    setIsPredicting(true);
    
    // Simulate API call delay
    setTimeout(() => {
      // Random prediction for demo (replace with actual API call)
      const isAir = Math.random() > 0.5;
      setPrediction(isAir ? "AIR" : "METAL");
      setConfidence(0.75 + Math.random() * 0.2);
      setIsPredicting(false);
      
      toast({
        title: "Classification Complete",
        description: `Predicted: ${isAir ? "AIR" : "METAL"}`,
      });
    }, 2000);
  };

  const signalCharts = parsedData ? [
    { name: "Current (kA)", dataIndex: 0, color: SIGNAL_COLORS["Current (kA)"] },
    { name: "Voltage (V)", dataIndex: 1, color: SIGNAL_COLORS["Voltage (V)"] },
    { name: "Resistance (Ohm)", dataIndex: 2, color: SIGNAL_COLORS["Resistance (Ohm)"] },
    { name: "Force (kg)", dataIndex: 3, color: SIGNAL_COLORS["Force (kg)"] },
  ] : [];

  return (
    <div className="min-h-screen py-8 px-4">
      {/* Background glow */}
      <div 
        className="fixed inset-0 pointer-events-none opacity-30"
        style={{ background: "radial-gradient(ellipse at 50% 0%, hsl(190 95% 50% / 0.1) 0%, transparent 50%)" }}
      />

      <div className="container max-w-6xl mx-auto relative">
        {/* Header */}
        <header className="text-center mb-12">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-secondary/50 border border-border/50 mb-6">
            <Activity className="w-4 h-4 text-primary" />
            <span className="text-sm text-muted-foreground font-mono">Signal Classification</span>
          </div>
          
          <h1 className="text-4xl md:text-5xl font-bold text-foreground mb-4 tracking-tight">
            AI Signal <span className="text-gradient-air">Classifier</span>
          </h1>
          
          <p className="text-muted-foreground max-w-xl mx-auto">
            Upload your Excel file with signal data to classify it as 
            <span className="text-air font-medium"> AIR</span> or 
            <span className="text-metal font-medium"> METAL</span> using our LSTM model.
          </p>
        </header>

        {/* Main Content Grid */}
        <div className="grid lg:grid-cols-2 gap-8">
          {/* Left Column - Upload & Result */}
          <div className="space-y-6">
            <FileUpload
              onFileSelect={handleFileSelect}
              isLoading={isLoading}
              isSuccess={isSuccess}
              error={error}
            />
            
            <PredictionResult
              prediction={prediction}
              confidence={confidence}
              isLoading={isPredicting}
            />
          </div>

          {/* Right Column - Data Preview */}
          <div className="space-y-6">
            {parsedData ? (
              <Tabs defaultValue="table" className="w-full">
                <TabsList className="grid w-full grid-cols-2 bg-secondary/50">
                  <TabsTrigger value="table" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">
                    <Table2 className="w-4 h-4 mr-2" />
                    Table View
                  </TabsTrigger>
                  <TabsTrigger value="charts" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">
                    <Activity className="w-4 h-4 mr-2" />
                    Signal Charts
                  </TabsTrigger>
                </TabsList>
                
                <TabsContent value="table" className="mt-4">
                  <DataTable
                    data={parsedData.rawData}
                    columns={parsedData.columns}
                    maxRows={15}
                  />
                </TabsContent>
                
                <TabsContent value="charts" className="mt-4 space-y-4">
                  {signalCharts.map((signal, i) => (
                    <SignalChart
                      key={signal.name}
                      data={parsedData.signalData}
                      signals={[signal]}
                      title={signal.name}
                    />
                  ))}
                </TabsContent>
              </Tabs>
            ) : (
              <div className="glass rounded-2xl p-12 text-center border-dashed">
                <Table2 className="w-12 h-12 text-muted-foreground mx-auto mb-4 opacity-50" />
                <p className="text-muted-foreground">
                  Data preview will appear here after upload
                </p>
              </div>
            )}
          </div>
        </div>

        {/* Backend Setup Info */}
        <div className="mt-12 glass rounded-2xl p-6">
          <div className="flex items-start gap-4">
            <div className="w-10 h-10 rounded-lg bg-primary/20 flex items-center justify-center flex-shrink-0">
              <Server className="w-5 h-5 text-primary" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-foreground mb-2">Backend Setup Required</h3>
              <p className="text-muted-foreground text-sm mb-4">
                This demo uses simulated predictions. To enable real classification, set up the Python Flask backend:
              </p>
              <div className="grid md:grid-cols-3 gap-4 text-sm">
                <div className="bg-secondary/50 rounded-lg p-4">
                  <p className="font-mono text-primary mb-1">1. Setup Environment</p>
                  <code className="text-xs text-muted-foreground">pip install flask tensorflow pandas scikit-learn</code>
                </div>
                <div className="bg-secondary/50 rounded-lg p-4">
                  <p className="font-mono text-primary mb-1">2. Place Model Files</p>
                  <code className="text-xs text-muted-foreground">model.keras, scaler.pkl in /models/</code>
                </div>
                <div className="bg-secondary/50 rounded-lg p-4">
                  <p className="font-mono text-primary mb-1">3. Run Server</p>
                  <code className="text-xs text-muted-foreground">python backend/app.py</code>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <footer className="mt-12 text-center text-sm text-muted-foreground">
          <p>LSTM Signal Classification • Built with React & Python</p>
        </footer>
      </div>
    </div>
  );
};

export default Index;
