import React, { useState, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { Upload, Video, Sparkles, Cloud } from 'lucide-react'
import axios from 'axios'

const UploadPage = () => {
  const [uploading, setUploading] = useState(false)
  const [dragOver, setDragOver] = useState(false)
  const navigate = useNavigate()

  const handleUpload = async (file) => {
    if (!file) return

    setUploading(true)
    
    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await axios.post('/api/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 300000, // 5 minutes timeout
      })

      if (response.data.video_id) {
        // Redirect to timeline page and start polling for status
        navigate(`/timeline/${response.data.video_id}`)
      }
    } catch (error) {
      console.error('Upload error:', error)
      alert(error.response?.data?.error || 'Upload failed. Please try again.')
    } finally {
      setUploading(false)
    }
  }

  const handleFileSelect = (event) => {
    const file = event.target.files[0]
    if (file) {
      handleUpload(file)
    }
  }

  const handleDrop = useCallback((event) => {
    event.preventDefault()
    setDragOver(false)
    
    const files = event.dataTransfer.files
    if (files.length > 0) {
      const file = files[0]
      if (file.type.startsWith('video/')) {
        handleUpload(file)
      } else {
        alert('Please upload a video file.')
      }
    }
  }, [])

  const handleDragOver = useCallback((event) => {
    event.preventDefault()
    setDragOver(true)
  }, [])

  const handleDragLeave = useCallback((event) => {
    event.preventDefault()
    setDragOver(false)
  }, [])

  return (
    <div className="upload-page">
      <div className="upload-container">
        <div className="header">
          <Sparkles size={52} color="#667eea" />
          <h1>Advanced AI Video Scene Detector</h1>
          <p>Upload any video and watch our AI automatically detect scenes, classify content, and generate intelligent tags</p>
          <p style={{ fontSize: '0.95rem', color: '#888' }}>
            Powered by custom-trained neural networks for accurate scene analysis
          </p>
        </div>

        <div
          className={`upload-box ${dragOver ? 'dragover' : ''}`}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onClick={() => document.getElementById('file-input').click()}
        >
          <div className="upload-icon">
            <Cloud size={64} />
          </div>
          <div className="upload-text">
            {dragOver ? '🎥 Drop video here!' : 'Click or drag & drop your video'}
          </div>
          <div className="upload-subtext">
            Supports MP4, AVI, MOV, MKV, WEBM files up to 500MB
            <br />
            Scene detection powered by advanced AI algorithms
          </div>
        </div>

        <input
          id="file-input"
          type="file"
          accept="video/*"
          onChange={handleFileSelect}
          style={{ display: 'none' }}
        />

        {uploading && (
          <div className="loading">
            <div className="spinner"></div>
            <div>
              <div style={{ fontWeight: '600', color: '#2d3748' }}>AI is processing your video...</div>
              <div style={{ fontSize: '0.9rem', color: '#718096', marginTop: '0.5rem' }}>
                Our neural network is analyzing scenes, classifying content, and generating tags
              </div>
              <div className="loading-dots" style={{ marginTop: '1rem' }}>
                <div></div>
                <div></div>
                <div></div>
              </div>
            </div>
          </div>
        )}

        <div style={{ marginTop: '2.5rem', color: '#718096', fontSize: '0.9rem', lineHeight: '1.6' }}>
          <strong>✨ AI Capabilities:</strong>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem', marginTop: '1rem' }}>
            <div>🎬 Scene Detection</div>
            <div>🏷️ Automatic Tagging</div>
            <div>📊 Content Classification</div>
            <div>🎥 Clip Extraction</div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default UploadPage