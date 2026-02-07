import { Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import Agents from './pages/Agents'
import Tools from './pages/Tools'
import ReasoningEngines from './pages/ReasoningEngines'
import Traces from './pages/Traces'
import Memory from './pages/Memory'
import Settings from './pages/Settings'

function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/agents" element={<Agents />} />
        <Route path="/tools" element={<Tools />} />
        <Route path="/reasoning" element={<ReasoningEngines />} />
        <Route path="/traces" element={<Traces />} />
        <Route path="/memory" element={<Memory />} />
        <Route path="/settings" element={<Settings />} />
      </Routes>
    </Layout>
  )
}

export default App
