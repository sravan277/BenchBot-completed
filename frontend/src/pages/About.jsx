import { motion } from 'framer-motion'
import { Target, BarChart3, Users, Code } from 'lucide-react'

export default function About() {
  return (
    <div className="min-h-screen py-12">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-12"
        >
          <h1 className="text-5xl font-bold text-white mb-4">About RAG Benchmark</h1>
          <p className="text-xl text-white/90">
            A comprehensive platform for evaluating Retrieval-Augmented Generation systems
          </p>
        </motion.div>

        <div className="space-y-8">
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="bg-white/90 backdrop-blur-md rounded-xl p-8 shadow-lg"
          >
            <div className="flex items-start space-x-4">
              <Target className="w-8 h-8 text-primary-600 flex-shrink-0 mt-1" />
              <div>
                <h2 className="text-2xl font-semibold text-gray-800 mb-3">Our Mission</h2>
                <p className="text-gray-600 leading-relaxed">
                  To provide researchers and developers with a standardized platform for evaluating
                  RAG pipelines across multiple dimensions including multilingual support, multi-hop
                  reasoning, and retrieval accuracy.
                </p>
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="bg-white/90 backdrop-blur-md rounded-xl p-8 shadow-lg"
          >
            <div className="flex items-start space-x-4">
              <BarChart3 className="w-8 h-8 text-primary-600 flex-shrink-0 mt-1" />
              <div>
                <h2 className="text-2xl font-semibold text-gray-800 mb-3">Evaluation Metrics</h2>
                <ul className="text-gray-600 space-y-2">
                  <li>• <strong>Precision:</strong> Measures the accuracy of retrieved documents</li>
                  <li>• <strong>Recall:</strong> Measures the completeness of retrieval</li>
                  <li>• <strong>F1 Score:</strong> Harmonic mean of precision and recall</li>
                  <li>• <strong>Similarity Score:</strong> Semantic similarity between retrieved and ground truth</li>
                  <li>• <strong>Latency:</strong> Query response time</li>
                </ul>
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="bg-white/90 backdrop-blur-md rounded-xl p-8 shadow-lg"
          >
            <div className="flex items-start space-x-4">
              <Code className="w-8 h-8 text-primary-600 flex-shrink-0 mt-1" />
              <div>
                <h2 className="text-2xl font-semibold text-gray-800 mb-3">Supported Features</h2>
                <ul className="text-gray-600 space-y-2">
                  <li>• Multiple embedding models (LaBSE, LEALLA, multilingual models)</li>
                  <li>• Customizable chunking strategies</li>
                  <li>• Multiple vector database backends (ChromaDB, FAISS)</li>
                  <li>• Support for English, Hindi, and Telugu languages</li>
                  <li>• Multi-hop query evaluation</li>
                </ul>
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="bg-white/90 backdrop-blur-md rounded-xl p-8 shadow-lg"
          >
            <div className="flex items-start space-x-4">
              <Users className="w-8 h-8 text-primary-600 flex-shrink-0 mt-1" />
              <div>
                <h2 className="text-2xl font-semibold text-gray-800 mb-3">Community</h2>
                <p className="text-gray-600 leading-relaxed">
                  Join our community of researchers and developers working on RAG systems.
                  Share your results, compare pipelines, and contribute to advancing the field
                  of retrieval-augmented generation.
                </p>
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  )
}

