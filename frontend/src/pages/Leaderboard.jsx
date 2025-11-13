import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import axios from 'axios'
import { motion } from 'framer-motion'
import { TrendingUp, ArrowUpDown, Eye, Search, Filter } from 'lucide-react'
import ResultsVisualization from '../components/ResultsVisualization'

export default function Leaderboard() {
  const [leaderboard, setLeaderboard] = useState([])
  const [filteredLeaderboard, setFilteredLeaderboard] = useState([])
  const [loading, setLoading] = useState(true)
  const [sortBy, setSortBy] = useState('overall_score')
  const [order, setOrder] = useState('desc')
  const [searchQuery, setSearchQuery] = useState('')
  const [filterByType, setFilterByType] = useState('all')
  const [selectedBenchmark, setSelectedBenchmark] = useState(null)
  const navigate = useNavigate()

  useEffect(() => {
    fetchLeaderboard()
  }, [sortBy, order])

  useEffect(() => {
    let filtered = leaderboard

    // Filter by search query
    if (searchQuery) {
      filtered = filtered.filter(
        (entry) =>
          entry.username.toLowerCase().includes(searchQuery.toLowerCase()) ||
          entry.benchmark_name?.toLowerCase().includes(searchQuery.toLowerCase())
      )
    }

    // Filter by benchmark type
    if (filterByType !== 'all') {
      filtered = filtered.filter((entry) =>
        entry.benchmark_types?.includes(filterByType)
      )
    }

    setFilteredLeaderboard(filtered)
  }, [leaderboard, searchQuery, filterByType])

  const fetchLeaderboard = async () => {
    try {
      setLoading(true)
      const response = await axios.get('/api/leaderboard', {
        params: { sort_by: sortBy, order, limit: 100 }
      })
      setLeaderboard(response.data.leaderboard)
      setFilteredLeaderboard(response.data.leaderboard)
    } catch (error) {
      console.error('Failed to fetch leaderboard:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleSort = (field) => {
    if (sortBy === field) {
      setOrder(order === 'desc' ? 'asc' : 'desc')
    } else {
      setSortBy(field)
      setOrder('desc')
    }
  }

  const handleViewDetails = async (benchmarkId) => {
    try {
      const response = await axios.get(`/api/leaderboard/${benchmarkId}`)
      setSelectedBenchmark(response.data)
    } catch (error) {
      console.error('Failed to fetch benchmark details:', error)
    }
  }

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-white text-xl">Loading leaderboard...</div>
      </div>
    )
  }

  if (selectedBenchmark) {
    return (
      <div className="min-h-screen py-12">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <button
            onClick={() => setSelectedBenchmark(null)}
            className="mb-6 px-4 py-2 bg-white/90 text-gray-700 rounded-lg hover:bg-white transition-colors"
          >
            ← Back to Leaderboard
          </button>
          <ResultsVisualization
            results={selectedBenchmark.results}
            benchmarkTypes={selectedBenchmark.benchmark_types}
          />
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen py-12">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-12"
        >
          <h1 className="text-5xl font-bold text-white mb-4">Leaderboard</h1>
          <p className="text-xl text-white/90">Compare RAG pipeline performance</p>
        </motion.div>

        {/* Search and Filter Controls */}
        <div className="bg-white/90 backdrop-blur-md rounded-xl p-6 mb-6 shadow-lg space-y-4">
          <div className="flex items-center space-x-4">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
              <input
                type="text"
                placeholder="Search by username or benchmark name..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500"
              />
            </div>
            <div className="flex items-center space-x-2">
              <Filter className="w-5 h-5 text-gray-600" />
              <select
                value={filterByType}
                onChange={(e) => setFilterByType(e.target.value)}
                className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500"
              >
                <option value="all">All Types</option>
                <option value="single">Single</option>
                <option value="multilingual">Multilingual</option>
                <option value="multi_hop">Multi-Hop</option>
              </select>
            </div>
          </div>
          <div className="flex items-center space-x-4 flex-wrap">
            <span className="text-gray-700 font-medium">Sort by:</span>
            {['overall_score', 'f1_score', 'precision', 'recall', 'similarity_score', 'latency'].map((field) => (
              <button
                key={field}
                onClick={() => handleSort(field)}
                className={`px-4 py-2 rounded-lg font-medium transition-all text-sm ${
                  sortBy === field
                    ? 'bg-primary-600 text-white'
                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                }`}
              >
                {field.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                {sortBy === field && (
                  <ArrowUpDown className="inline w-4 h-4 ml-2" />
                )}
              </button>
            ))}
          </div>
        </div>

        {/* Results Count */}
        <div className="mb-4 text-white text-sm">
          Showing {filteredLeaderboard.length} of {leaderboard.length} benchmarks
        </div>

        {/* Leaderboard Table */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white/90 backdrop-blur-md rounded-xl shadow-lg overflow-hidden"
        >
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-primary-600 text-white">
                <tr>
                  <th className="px-6 py-4 text-left font-semibold">Rank</th>
                  <th className="px-6 py-4 text-left font-semibold">User</th>
                  <th className="px-6 py-4 text-left font-semibold">Benchmark Name</th>
                  <th className="px-6 py-4 text-left font-semibold">Types</th>
                  <th className="px-6 py-4 text-left font-semibold">Overall Score</th>
                  <th className="px-6 py-4 text-left font-semibold">F1 Score</th>
                  <th className="px-6 py-4 text-left font-semibold">Precision</th>
                  <th className="px-6 py-4 text-left font-semibold">Recall</th>
                  <th className="px-6 py-4 text-left font-semibold">Similarity</th>
                  <th className="px-6 py-4 text-left font-semibold">Latency (ms)</th>
                  <th className="px-6 py-4 text-left font-semibold">Actions</th>
                </tr>
              </thead>
              <tbody>
                {filteredLeaderboard.map((entry, index) => {
                  // Get metrics from first benchmark type
                  const firstType = entry.benchmark_types?.[0] || 'single'
                  const metrics = entry.metrics?.[firstType] || {}
                  
                  return (
                    <tr
                      key={entry.id}
                      className="border-b border-gray-200 hover:bg-gray-50 transition-colors"
                    >
                      <td className="px-6 py-4">
                        <div className="flex items-center">
                          {index < 3 && (
                            <TrendingUp className="w-5 h-5 text-yellow-500 mr-2" />
                          )}
                          <span className="font-semibold">#{index + 1}</span>
                        </div>
                      </td>
                      <td className="px-6 py-4 font-medium text-gray-800">{entry.username}</td>
                      <td className="px-6 py-4 font-medium text-primary-600">{entry.benchmark_name || 'Unnamed Benchmark'}</td>
                      <td className="px-6 py-4">
                        <div className="flex flex-wrap gap-1">
                          {entry.benchmark_types?.map((type) => (
                            <span
                              key={type}
                              className="px-2 py-1 text-xs bg-primary-100 text-primary-700 rounded"
                            >
                              {type.replace('_', ' ')}
                            </span>
                          ))}
                        </div>
                      </td>
                      <td className="px-6 py-4">
                        <span className="font-bold text-primary-600">
                          {(entry.overall_score * 100).toFixed(2)}%
                        </span>
                      </td>
                      <td className="px-6 py-4">{(metrics.f1_score * 100 || 0).toFixed(2)}%</td>
                      <td className="px-6 py-4">{(metrics.precision * 100 || 0).toFixed(2)}%</td>
                      <td className="px-6 py-4">{(metrics.recall * 100 || 0).toFixed(2)}%</td>
                      <td className="px-6 py-4">{(metrics.similarity_score * 100 || 0).toFixed(2)}%</td>
                      <td className="px-6 py-4">{(metrics.latency * 1000 || 0).toFixed(2)}</td>
                      <td className="px-6 py-4">
                        <button
                          onClick={() => handleViewDetails(entry.id)}
                          className="flex items-center space-x-2 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors"
                        >
                          <Eye className="w-4 h-4" />
                          <span>View</span>
                        </button>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </motion.div>

        {filteredLeaderboard.length === 0 && leaderboard.length > 0 && (
          <div className="text-center py-12 text-white text-xl">
            No benchmarks match your search criteria.
          </div>
        )}
        {leaderboard.length === 0 && (
          <div className="text-center py-12 text-white text-xl">
            No benchmarks found. Be the first to run a benchmark!
          </div>
        )}
      </div>
    </div>
  )
}

