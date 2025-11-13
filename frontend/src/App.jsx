import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { AuthProvider } from './contexts/AuthContext'
import { ToastProvider } from './contexts/ToastContext'
import Navbar from './components/Navbar'
import Home from './pages/Home'
import Benchmark from './pages/Benchmark'
import Leaderboard from './pages/Leaderboard'
import About from './pages/About'
import Login from './pages/Login'
import Signup from './pages/Signup'

function App() {
  return (
    <ToastProvider>
      <AuthProvider>
        <Router>
          <div className="min-h-screen">
            <Navbar />
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/benchmark" element={<Benchmark />} />
              <Route path="/leaderboard" element={<Leaderboard />} />
              <Route path="/about" element={<About />} />
              <Route path="/login" element={<Login />} />
              <Route path="/signup" element={<Signup />} />
            </Routes>
          </div>
        </Router>
      </AuthProvider>
    </ToastProvider>
  )
}

export default App

