import { useState, useEffect, useRef } from 'react'
import { useAuth } from '../contexts/AuthContext'
import { useToast } from '../contexts/ToastContext'
import { useNavigate } from 'react-router-dom'
import axios from 'axios'
import { motion } from 'framer-motion'
import { Upload, Play, CheckCircle, AlertCircle, Loader, Download } from 'lucide-react'
import ResultsVisualization from '../components/ResultsVisualization'

const EMBEDDING_MODELS = [
  { value: 'sentence-transformers/LaBSE', label: 'LaBSE (Multilingual)' },
  { value: 'setu4993/LEALLA-base', label: 'LEALLA-base (Multilingual)' },
  { value: 'intfloat/multilingual-e5-small', label: 'Multilingual-E5-Small (Multilingual)' },
  { value: 'sentence-transformers/use-cmlm-multilingual', label: 'USE-CMLM (Multilingual)' },
  { value: 'sentence-transformers/distiluse-base-multilingual-cased-v2', label: 'DistilUSE (Multilingual)' },
  { value: 'sentence-transformers/all-MiniLM-L6-v2', label: 'MiniLM-L6-v2 (English)' }
]

const VECTOR_DBS = [
  { value: 'chroma', label: 'ChromaDB' },
  { value: 'faiss', label: 'FAISS' },
  { value: 'qdrant', label: 'Qdrant (Cloud)' }
]

export default function Benchmark() {
  const { user } = useAuth()
  const { showToast } = useToast()
  const navigate = useNavigate()
  const [step, setStep] = useState(1) // 1: config, 2: benchmark selection, 3: results
  const [configType, setConfigType] = useState('prebuilt') // 'prebuilt' or 'upload'
  
  // Prebuilt config
  const [chunkSize, setChunkSize] = useState(500)
  const [chunkOverlap, setChunkOverlap] = useState(50)
  const [embeddingModel, setEmbeddingModel] = useState(EMBEDDING_MODELS[0].value)
  const [vectorDb, setVectorDb] = useState('chroma')
  const [topK, setTopK] = useState(5)
  
  // Upload
  const [pipelineFile, setPipelineFile] = useState(null)
  const [pipelineConfig, setPipelineConfig] = useState(null)
  
  // Benchmark selection
  const [selectedBenchmarks, setSelectedBenchmarks] = useState([])
  const [benchmarkName, setBenchmarkName] = useState('')
  
  // Results
  const [results, setResults] = useState(null)
  const [benchmarkId, setBenchmarkId] = useState(null)
  const [loading, setLoading] = useState(false)
  const [progress, setProgress] = useState('')
  const [progressPercent, setProgressPercent] = useState(0)
  const [currentStep, setCurrentStep] = useState('')
  const [jobId, setJobId] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    if (!user) {
      navigate('/login')
    }
  }, [user, navigate])

  const handleFileUpload = async (e) => {
    const file = e.target.files[0]
    if (!file) return

    setPipelineFile(file)
    setError('')

    try {
      const formData = new FormData()
      formData.append('file', file)

      const response = await axios.post('/api/benchmark/upload-pipeline', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      })

      setPipelineConfig(response.data.pipeline_config)
      showToast('Pipeline configuration uploaded successfully!', 'success')
    } catch (err) {
      const errorMsg = err.response?.data?.detail || 'Failed to upload pipeline file'
      setError(errorMsg)
      showToast(errorMsg, 'error')
      setPipelineFile(null)
    }
  }

  const handleNext = () => {
    if (step === 1) {
      if (configType === 'upload' && !pipelineConfig) {
        setError('Please upload a pipeline JSON file')
        return
      }
      setStep(2)
    }
  }

  const handleBenchmarkToggle = (benchmark) => {
    setSelectedBenchmarks(prev =>
      prev.includes(benchmark)
        ? prev.filter(b => b !== benchmark)
        : [...prev, benchmark]
    )
  }

  const intervalRef = useRef(null)

  // Poll progress
  useEffect(() => {
    if (!jobId || !loading) {
      // Clear interval if jobId or loading changes
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
        intervalRef.current = null
      }
      return
    }

    let isMounted = true
    let shouldStop = false

    const pollProgress = async () => {
      if (!isMounted || shouldStop || !jobId) {
        if (intervalRef.current) {
          clearInterval(intervalRef.current)
          intervalRef.current = null
        }
        return
      }
      
      try {
        const response = await axios.get(`/api/benchmark/progress/${jobId}`)
        const data = response.data

        if (!isMounted || shouldStop || !jobId) {
          if (intervalRef.current) {
            clearInterval(intervalRef.current)
            intervalRef.current = null
          }
          return
        }

        setProgressPercent(data.progress || 0)
        setCurrentStep(data.current_step || '')
        setProgress(data.details || '')

        if (data.status === 'completed') {
          shouldStop = true
          if (intervalRef.current) {
            clearInterval(intervalRef.current)
            intervalRef.current = null
          }
          setResults(data.results)
          setBenchmarkId(data.benchmark_id)
          setStep(3)
          setLoading(false)
          setJobId(null)
          showToast('Benchmark completed successfully!', 'success')
        } else if (data.status === 'error') {
          shouldStop = true
          if (intervalRef.current) {
            clearInterval(intervalRef.current)
            intervalRef.current = null
          }
          setError(data.error || 'An error occurred')
          setLoading(false)
          setJobId(null)
          showToast(data.error || 'An error occurred', 'error')
        }
      } catch (err) {
        console.error('Error polling progress:', err)
        // If job not found, stop polling
        if (err.response?.status === 404) {
          shouldStop = true
          if (intervalRef.current) {
            clearInterval(intervalRef.current)
            intervalRef.current = null
          }
          setLoading(false)
          setJobId(null)
        }
      }
    }

    intervalRef.current = setInterval(pollProgress, 1000) // Poll every second
    pollProgress() // Initial poll

    return () => {
      shouldStop = true
      isMounted = false
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
        intervalRef.current = null
      }
    }
  }, [jobId, loading])

  const handleRunBenchmark = async () => {
    if (selectedBenchmarks.length === 0) {
      const errorMsg = 'Please select at least one benchmark type'
      setError(errorMsg)
      showToast(errorMsg, 'error')
      return
    }

    // Validate inputs
    if (chunkSize < 50 || chunkSize > 5000) {
      const errorMsg = 'Chunk size must be between 50 and 5000'
      setError(errorMsg)
      showToast(errorMsg, 'error')
      return
    }

    if (chunkOverlap < 0 || chunkOverlap >= chunkSize) {
      const errorMsg = 'Chunk overlap must be between 0 and less than chunk size'
      setError(errorMsg)
      showToast(errorMsg, 'error')
      return
    }

    if (topK < 1 || topK > 50) {
      const errorMsg = 'Top K must be between 1 and 50'
      setError(errorMsg)
      showToast(errorMsg, 'error')
      return
    }

    setLoading(true)
    setError('')
    setProgress('Starting benchmark...')
    setProgressPercent(0)
    setCurrentStep('Initializing')

    try {
      const requestData = {
        benchmark_types: selectedBenchmarks,
        benchmark_name: benchmarkName || undefined
      }

      if (configType === 'prebuilt') {
        requestData.config = {
          chunk_size: chunkSize,
          chunk_overlap: chunkOverlap,
          embedding_model: embeddingModel,
          vector_db: vectorDb,
          reranking_strategy: 'none',
          top_k: topK
        }
      } else {
        requestData.pipeline_json = pipelineConfig
      }

      const response = await axios.post('/api/benchmark/run', requestData)
      setJobId(response.data.job_id)
      showToast('Benchmark started!', 'success')
    } catch (err) {
      const errorMsg = err.response?.data?.detail || err.message || 'Failed to start benchmark'
      setError(errorMsg)
      showToast(errorMsg, 'error')
      setLoading(false)
      setJobId(null)
    }
  }

  const handleExportResults = () => {
    if (!results) return

    const exportData = {
      benchmark_name: benchmarkName || 'Unnamed Benchmark',
      benchmark_id: benchmarkId,
      timestamp: new Date().toISOString(),
      results: results
    }

    const dataStr = JSON.stringify(exportData, null, 2)
    const dataBlob = new Blob([dataStr], { type: 'application/json' })
    const url = URL.createObjectURL(dataBlob)
    const link = document.createElement('a')
    link.href = url
    link.download = `benchmark_results_${benchmarkName || Date.now()}.json`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    URL.revokeObjectURL(url)
    showToast('Results exported successfully!', 'success')
  }

  if (!user) return null

  return (
    <div className="min-h-screen py-12">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-12"
        >
          <h1 className="text-5xl font-bold text-white mb-4">RAG Benchmark</h1>
          <p className="text-xl text-white/90">Configure and evaluate your RAG pipeline</p>
        </motion.div>

        {/* Progress Steps */}
        <div className="flex justify-center mb-8">
          <div className="flex items-center space-x-4">
            {[1, 2, 3].map((s) => (
              <div key={s} className="flex items-center">
                <div
                  className={`w-10 h-10 rounded-full flex items-center justify-center font-semibold ${
                    step >= s
                      ? 'bg-primary-600 text-white'
                      : 'bg-white/20 text-white/50'
                  }`}
                >
                  {s}
                </div>
                {s < 3 && (
                  <div
                    className={`w-16 h-1 ${
                      step > s ? 'bg-primary-600' : 'bg-white/20'
                    }`}
                  />
                )}
              </div>
            ))}
          </div>
        </div>

        {error && (
          <div className="mb-6 p-4 bg-red-100 border border-red-400 text-red-700 rounded-lg flex items-center space-x-2">
            <AlertCircle className="w-5 h-5" />
            <span>{error}</span>
          </div>
        )}

        {/* Step 1: Configuration */}
        {step === 1 && (
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="bg-white/90 backdrop-blur-md rounded-xl p-8 shadow-lg"
          >
            <h2 className="text-2xl font-semibold text-gray-800 mb-6">Pipeline Configuration</h2>

            <div className="mb-6">
              <div className="flex space-x-4 mb-4">
                <button
                  onClick={() => setConfigType('prebuilt')}
                  className={`flex-1 py-3 px-4 rounded-lg font-semibold transition-all ${
                    configType === 'prebuilt'
                      ? 'bg-primary-600 text-white'
                      : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                  }`}
                >
                  Prebuilt Configuration
                </button>
                <button
                  onClick={() => setConfigType('upload')}
                  className={`flex-1 py-3 px-4 rounded-lg font-semibold transition-all ${
                    configType === 'upload'
                      ? 'bg-primary-600 text-white'
                      : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                  }`}
                >
                  Upload JSON
                </button>
              </div>
            </div>

            {configType === 'prebuilt' ? (
              <div className="space-y-6">
                <div className="grid md:grid-cols-2 gap-6">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Chunk Size (50-5000)
                    </label>
                    <input
                      type="number"
                      value={chunkSize}
                      onChange={(e) => {
                        const val = parseInt(e.target.value) || 500
                        setChunkSize(Math.max(50, Math.min(5000, val)))
                      }}
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500"
                      min="50"
                      max="5000"
                    />
                    <p className="mt-1 text-xs text-gray-500">Recommended: 256-1024</p>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Chunk Overlap (0-{Math.min(chunkSize - 1, 500)})
                    </label>
                    <input
                      type="number"
                      value={chunkOverlap}
                      onChange={(e) => {
                        const val = parseInt(e.target.value) || 0
                        setChunkOverlap(Math.max(0, Math.min(chunkSize - 1, val)))
                      }}
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500"
                      min="0"
                      max={chunkSize - 1}
                    />
                    <p className="mt-1 text-xs text-gray-500">Recommended: 10-20% of chunk size</p>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Embedding Model
                    </label>
                    <select
                      value={embeddingModel}
                      onChange={(e) => setEmbeddingModel(e.target.value)}
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500"
                    >
                      {EMBEDDING_MODELS.map((model) => (
                        <option key={model.value} value={model.value}>
                          {model.label}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Vector Database
                    </label>
                    <select
                      value={vectorDb}
                      onChange={(e) => setVectorDb(e.target.value)}
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500"
                    >
                      {VECTOR_DBS.map((db) => (
                        <option key={db.value} value={db.value}>
                          {db.label}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Top K Results (1-50)
                    </label>
                    <input
                      type="number"
                      value={topK}
                      onChange={(e) => {
                        const val = parseInt(e.target.value) || 5
                        setTopK(Math.max(1, Math.min(50, val)))
                      }}
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500"
                      min="1"
                      max="50"
                    />
                    <p className="mt-1 text-xs text-gray-500">Number of top results to retrieve</p>
                  </div>
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Upload Pipeline JSON
                </label>
                <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center">
                  <Upload className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                  <input
                    type="file"
                    accept=".json"
                    onChange={handleFileUpload}
                    className="hidden"
                    id="pipeline-upload"
                  />
                  <label
                    htmlFor="pipeline-upload"
                    className="cursor-pointer inline-block px-6 py-3 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors"
                  >
                    Choose File
                  </label>
                  {pipelineFile && (
                    <p className="mt-4 text-sm text-gray-600">
                      {pipelineFile.name} <CheckCircle className="inline w-4 h-4 text-green-500" />
                    </p>
                  )}
                </div>
              </div>
            )}

            <div className="mt-8 flex justify-end">
              <button
                onClick={handleNext}
                className="px-6 py-3 bg-primary-600 text-white rounded-lg font-semibold hover:bg-primary-700 transition-colors"
              >
                Next
              </button>
            </div>
          </motion.div>
        )}

        {/* Step 2: Benchmark Selection */}
        {step === 2 && (
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="bg-white/90 backdrop-blur-md rounded-xl p-8 shadow-lg"
          >
            <h2 className="text-2xl font-semibold text-gray-800 mb-6">Select Benchmark Types</h2>

            {/* Benchmark Name Input */}
            <div className="mb-6">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Benchmark Name (Optional)
              </label>
              <input
                type="text"
                value={benchmarkName}
                onChange={(e) => setBenchmarkName(e.target.value)}
                placeholder="e.g., My RAG Pipeline v1.0"
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              />
              <p className="mt-1 text-sm text-gray-500">
                Give your benchmark a name to easily identify it in the leaderboard
              </p>
            </div>

            <div className="space-y-4 mb-8">
              {[
                { id: 'single', label: 'Single Document', desc: 'Evaluate on single document queries' },
                { id: 'multilingual', label: 'Multilingual', desc: 'Test across English, Hindi, and Telugu' },
                { id: 'multi_hop', label: 'Multi-Hop', desc: 'Evaluate queries requiring multiple documents' }
              ].map((benchmark) => (
                <div
                  key={benchmark.id}
                  onClick={() => handleBenchmarkToggle(benchmark.id)}
                  className={`p-4 border-2 rounded-lg cursor-pointer transition-all ${
                    selectedBenchmarks.includes(benchmark.id)
                      ? 'border-primary-600 bg-primary-50'
                      : 'border-gray-300 hover:border-primary-300'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="font-semibold text-gray-800">{benchmark.label}</h3>
                      <p className="text-sm text-gray-600">{benchmark.desc}</p>
                    </div>
                    {selectedBenchmarks.includes(benchmark.id) && (
                      <CheckCircle className="w-6 h-6 text-primary-600" />
                    )}
                  </div>
                </div>
              ))}
            </div>

            <div className="flex justify-between">
              <button
                onClick={() => setStep(1)}
                className="px-6 py-3 bg-gray-300 text-gray-700 rounded-lg font-semibold hover:bg-gray-400 transition-colors"
              >
                Back
              </button>
              <button
                onClick={handleRunBenchmark}
                disabled={loading || selectedBenchmarks.length === 0}
                className="px-6 py-3 bg-primary-600 text-white rounded-lg font-semibold hover:bg-primary-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2"
              >
                {loading ? (
                  <>
                    <Loader className="w-5 h-5 animate-spin" />
                    <span>Running Benchmark...</span>
                  </>
                ) : (
                  <>
                    <Play className="w-5 h-5" />
                    <span>Start Benchmarking</span>
                  </>
                )}
              </button>
            </div>
            {loading && (
              <div className="mt-6 p-6 bg-gradient-to-r from-blue-50 to-indigo-50 border-2 border-blue-200 rounded-xl shadow-lg">
                <div className="mb-4">
                  <div className="flex items-center justify-between mb-2">
                    <h3 className="text-lg font-semibold text-gray-800">{currentStep || 'Running Benchmark'}</h3>
                    <span className="text-sm font-medium text-primary-600">{Math.round(progressPercent)}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
                    <div
                      className="bg-gradient-to-r from-primary-500 to-primary-600 h-3 rounded-full transition-all duration-300 ease-out"
                      style={{ width: `${progressPercent}%` }}
                    />
                  </div>
                </div>
                <div className="mt-4 p-3 bg-white/80 rounded-lg border border-blue-100">
                  <div className="flex items-start space-x-3">
                    <Loader className="w-5 h-5 text-primary-600 animate-spin mt-0.5 flex-shrink-0" />
                    <div className="flex-1">
                      <p className="text-sm text-gray-700 font-medium">{progress || 'Processing...'}</p>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </motion.div>
        )}

        {/* Step 3: Results */}
        {step === 3 && results && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <div className="mb-6 flex justify-between items-center">
              <h2 className="text-2xl font-bold text-white">Benchmark Results</h2>
              <button
                onClick={handleExportResults}
                className="flex items-center space-x-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
              >
                <Download className="w-4 h-4" />
                <span>Export Results</span>
              </button>
            </div>
            <ResultsVisualization results={results} benchmarkTypes={selectedBenchmarks} />
            <div className="mt-8 flex justify-center space-x-4">
              <button
                onClick={() => {
                  setStep(1)
                  setResults(null)
                  setSelectedBenchmarks([])
                  setBenchmarkName('')
                  setBenchmarkId(null)
                }}
                className="px-6 py-3 bg-gray-300 text-gray-700 rounded-lg font-semibold hover:bg-gray-400 transition-colors"
              >
                Run Another Benchmark
              </button>
              <button
                onClick={() => navigate('/leaderboard')}
                className="px-6 py-3 bg-primary-600 text-white rounded-lg font-semibold hover:bg-primary-700 transition-colors"
              >
                View Leaderboard
              </button>
            </div>
          </motion.div>
        )}
      </div>
    </div>
  )
}

