import { motion } from 'framer-motion'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  LineChart,
  Line,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  PieChart,
  Pie,
  Cell
} from 'recharts'
import { TrendingUp, Clock, Target, Zap } from 'lucide-react'

const COLORS = ['#0ea5e9', '#8b5cf6', '#ec4899', '#10b981', '#f59e0b']

export default function ResultsVisualization({ results, benchmarkTypes }) {
  const prepareBarData = (benchmarkType) => {
    if (!results[benchmarkType]) return []
    const metrics = results[benchmarkType].overall_metrics
    return [
      { name: 'Precision', value: (metrics.precision * 100).toFixed(2) },
      { name: 'Recall', value: (metrics.recall * 100).toFixed(2) },
      { name: 'F1 Score', value: (metrics.f1_score * 100).toFixed(2) },
      { name: 'Similarity', value: (metrics.similarity_score * 100).toFixed(2) }
    ]
  }

  const prepareRadarData = (benchmarkType) => {
    if (!results[benchmarkType]) return []
    const metrics = results[benchmarkType].overall_metrics
    return [
      {
        metric: 'Precision',
        value: metrics.precision * 100
      },
      {
        metric: 'Recall',
        value: metrics.recall * 100
      },
      {
        metric: 'F1 Score',
        value: metrics.f1_score * 100
      },
      {
        metric: 'Similarity',
        value: metrics.similarity_score * 100
      }
    ]
  }

  const prepareLanguageComparison = () => {
    if (!results.multilingual?.language_metrics) return []
    const langMetrics = results.multilingual.language_metrics
    return Object.keys(langMetrics).map(lang => ({
      language: lang === 'en' ? 'English' : lang === 'hi' ? 'Hindi' : 'Telugu',
      precision: langMetrics[lang].precision * 100,
      recall: langMetrics[lang].recall * 100,
      f1_score: langMetrics[lang].f1_score * 100,
      similarity: langMetrics[lang].similarity_score * 100
    }))
  }

  return (
    <div className="space-y-8">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white/90 backdrop-blur-md rounded-xl p-8 shadow-lg"
      >
        <h2 className="text-3xl font-bold text-gray-800 mb-6">Benchmark Results</h2>

        {/* Overall Metrics Cards */}
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          {benchmarkTypes.map((type) => {
            if (!results[type]) return null
            const metrics = results[type].overall_metrics
            const overallScore = (
              metrics.f1_score * 0.4 +
              metrics.similarity_score * 0.3 +
              metrics.precision * 0.2 +
              metrics.recall * 0.1
            ) * 100
            return (
              <motion.div
                key={type}
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                className="bg-gradient-to-br from-primary-50 to-purple-50 p-6 rounded-lg shadow-lg hover:shadow-xl transition-shadow"
              >
                <h3 className="text-lg font-semibold text-gray-700 mb-2 capitalize">{type.replace('_', ' ')}</h3>
                <div className="mb-4">
                  <div className="text-3xl font-bold text-primary-600">{overallScore.toFixed(1)}%</div>
                  <div className="text-xs text-gray-500">Overall Score</div>
                </div>
                <div className="space-y-2 text-sm">
                  <div className="flex items-center justify-between">
                    <span className="text-gray-600">F1 Score</span>
                    <span className="font-bold text-primary-600">{(metrics.f1_score * 100).toFixed(2)}%</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-gray-600">Precision</span>
                    <span className="font-bold text-primary-600">{(metrics.precision * 100).toFixed(2)}%</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-gray-600">Recall</span>
                    <span className="font-bold text-primary-600">{(metrics.recall * 100).toFixed(2)}%</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-gray-600">Similarity</span>
                    <span className="font-bold text-primary-600">{(metrics.similarity_score * 100).toFixed(2)}%</span>
                  </div>
                  <div className="flex items-center justify-between pt-2 border-t">
                    <span className="text-gray-600">Latency</span>
                    <span className="font-bold text-primary-600">{(metrics.latency * 1000).toFixed(2)}ms</span>
                  </div>
                </div>
              </motion.div>
            )
          })}
        </div>

        {/* Bar Charts */}
        <div className="grid md:grid-cols-2 gap-6 mb-8">
          {benchmarkTypes.map((type) => {
            const data = prepareBarData(type)
            if (data.length === 0) return null
            return (
              <div key={type} className="bg-white p-6 rounded-lg shadow">
                <h3 className="text-xl font-semibold text-gray-800 mb-4 capitalize">
                  {type.replace('_', ' ')} - Metrics
                </h3>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={data}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="value" fill="#0ea5e9" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )
          })}
        </div>

        {/* Radar Charts */}
        <div className="grid md:grid-cols-2 gap-6 mb-8">
          {benchmarkTypes.map((type) => {
            const data = prepareRadarData(type)
            if (data.length === 0) return null
            return (
              <div key={type} className="bg-white p-6 rounded-lg shadow">
                <h3 className="text-xl font-semibold text-gray-800 mb-4 capitalize">
                  {type.replace('_', ' ')} - Performance Radar
                </h3>
                <ResponsiveContainer width="100%" height={300}>
                  <RadarChart data={data}>
                    <PolarGrid />
                    <PolarAngleAxis dataKey="metric" />
                    <PolarRadiusAxis angle={90} domain={[0, 100]} />
                    <Radar
                      name="Performance"
                      dataKey="value"
                      stroke="#0ea5e9"
                      fill="#0ea5e9"
                      fillOpacity={0.6}
                    />
                  </RadarChart>
                </ResponsiveContainer>
              </div>
            )
          })}
        </div>

        {/* Language Comparison for Multilingual */}
        {results.multilingual && results.multilingual.language_metrics && (
          <div className="bg-white p-6 rounded-lg shadow">
            <h3 className="text-xl font-semibold text-gray-800 mb-4">Multilingual Performance Comparison</h3>
            <ResponsiveContainer width="100%" height={400}>
              <BarChart data={prepareLanguageComparison()}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="language" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="precision" fill="#0ea5e9" name="Precision" />
                <Bar dataKey="recall" fill="#8b5cf6" name="Recall" />
                <Bar dataKey="f1_score" fill="#ec4899" name="F1 Score" />
                <Bar dataKey="similarity" fill="#10b981" name="Similarity" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </motion.div>
    </div>
  )
}

