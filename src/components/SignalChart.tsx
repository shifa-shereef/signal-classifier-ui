import { useMemo } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";

interface SignalChartProps {
  data: number[][];
  signals: {
    name: string;
    dataIndex: number;
    color: string;
  }[];
  title?: string;
}

export function SignalChart({ data, signals, title }: SignalChartProps) {
  const chartData = useMemo(() => {
    // Sample data for better performance (max 200 points)
    const step = Math.max(1, Math.floor(data.length / 200));
    return data
      .filter((_, i) => i % step === 0)
      .map((row, index) => {
        const point: Record<string, number> = { index: index * step };
        signals.forEach(signal => {
          point[signal.name] = row[signal.dataIndex] || 0;
        });
        return point;
      });
  }, [data, signals]);

  return (
    <div className="glass rounded-xl p-4">
      {title && (
        <h3 className="text-sm font-medium text-foreground mb-4">{title}</h3>
      )}
      <div className="h-[200px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={chartData}
            margin={{ top: 5, right: 10, left: 0, bottom: 5 }}
          >
            <CartesianGrid 
              strokeDasharray="3 3" 
              stroke="hsl(222 30% 18%)" 
              vertical={false}
            />
            <XAxis 
              dataKey="index" 
              stroke="hsl(215 20% 55%)"
              fontSize={10}
              tickLine={false}
              axisLine={false}
            />
            <YAxis 
              stroke="hsl(215 20% 55%)"
              fontSize={10}
              tickLine={false}
              axisLine={false}
              width={50}
              tickFormatter={(value) => value.toFixed(2)}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "hsl(222 47% 9%)",
                border: "1px solid hsl(222 30% 18%)",
                borderRadius: "8px",
                fontSize: "12px",
              }}
              labelStyle={{ color: "hsl(215 20% 55%)" }}
            />
            <Legend 
              wrapperStyle={{ fontSize: "11px" }}
              iconType="line"
            />
            {signals.map(signal => (
              <Line
                key={signal.name}
                type="monotone"
                dataKey={signal.name}
                stroke={signal.color}
                strokeWidth={1.5}
                dot={false}
                activeDot={{ r: 4, fill: signal.color }}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
