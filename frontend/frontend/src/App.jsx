import React from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import UploadPage from './components/UploadPage'
import TimelinePage from './components/TimelinePage'
import './App.css'

function App() {
  return (
    <Router>
      <div className="App">
        <Routes>
          <Route path="/" element={<UploadPage />} />
          <Route path="/timeline/:videoId" element={<TimelinePage />} />
        </Routes>
      </div>
    </Router>
  )
}

export default App
