import React, { useState, useRef, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { Play, Pause, Search, Clock, Tag, ArrowLeft, Video, Sparkles, BarChart3 } from 'lucide-react'
import axios from 'axios'

const TimelinePage = () => {
  const { videoId } = useParams()
  const navigate = useNavigate()
  const videoRef = useRef(null)
  const [videoData, setVideoData] = useState(null)
  const [currentTime, setCurrentTime] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [activeScene, setActiveScene] = useState(null)
  const [playingClip, setPlayingClip] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  // Poll for video processing status
  useEffect(() => {
    const pollStatus = async () => {
      try {
        const response = await axios.get(`/api/status/${videoId}`)
        const data = response.data
        
        if (data.status === 'completed') {
          setVideoData(data)
          setLoading(false)
          if (data.scenes && data.scenes.length > 0) {
            setActiveScene(data.scenes[0])
          }
        } else if (data.status === 'processing') {
          // Continue polling every 2 seconds
          setTimeout(pollStatus, 2000)
        } else if (data.status === 'failed') {
          setError(data.error || 'Processing failed')
          setLoading(false)
        }
      } catch (err) {
        setError('Failed to fetch video status')
        setLoading(false)
      }
    }

    pollStatus()
  }, [videoId])

  // Auto-navigate to first matching scene when searching
  useEffect(() => {
    if (searchQuery && filteredScenes.length > 0) {
      const firstMatchingScene = filteredScenes[0]
      seekToTime(firstMatchingScene.start_time)
      setActiveScene(firstMatchingScene)
    }
  }, [searchQuery])

  // Video time update listener
  useEffect(() => {
    const video = videoRef.current
    if (!video) return

    const updateTime = () => setCurrentTime(video.currentTime)
    video.addEventListener('timeupdate', updateTime)
    
    return () => video.removeEventListener('timeupdate', updateTime)
  }, [])

  // Find active scene based on current time
  useEffect(() => {
    if (videoData?.scenes) {
      const currentScene = videoData.scenes.find(scene => 
        currentTime >= scene.start_time && currentTime <= scene.end_time
      )
      if (currentScene) {
        setActiveScene(currentScene)
      }
    }
  }, [currentTime, videoData])

  const handlePlayPause = () => {
    const video = videoRef.current
    if (!video) return

    if (video.paused) {
      video.play()
      setIsPlaying(true)
    } else {
      video.pause()
      setIsPlaying(false)
    }
  }

  const seekToTime = (time) => {
    const video = videoRef.current
    if (video) {
      video.currentTime = time
      setCurrentTime(time)
      video.play().then(() => setIsPlaying(true))
    }
  }

  const handleTagClick = (tag) => {
    setSearchQuery(tag)
  }

  const playClip = async (scene) => {
    if (scene.clip_path) {
      setPlayingClip({ scene, clipUrl: `/api/clips/${videoId}/${scene.scene_id}.mp4` })
    }
  }

  const stopClip = () => {
    setPlayingClip(null)
  }

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  const filteredScenes = videoData?.scenes?.filter(scene =>
    scene.tags.some(tag =>
      tag.toLowerCase().includes(searchQuery.toLowerCase())
    ) || scene.scene_class.toLowerCase().includes(searchQuery.toLowerCase())
  ) || []

  if (loading) {
    return (
      <div className="timeline-page">
        <div className="loading" style={{ minHeight: '60vh', flexDirection: 'column' }}>
          <div className="spinner"></div>
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontWeight: '600', fontSize: '1.2rem', color: '#2d3748', marginBottom: '0.5rem' }}>
              AI is analyzing your video...
            </div>
            <div style={{ color: '#718096' }}>
              Detecting scenes, classifying content, and generating tags
            </div>
            <div className="loading-dots" style={{ marginTop: '1rem' }}>
              <div></div>
              <div></div>
              <div></div>
            </div>
          </div>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="timeline-page">
        <div className="error-message">
          <h3>Processing Error</h3>
          <p>{error}</p>
          <button 
            className="btn" 
            onClick={() => navigate('/')}
            style={{ marginTop: '1rem' }}
          >
            <ArrowLeft size={16} />
            Back to Upload
          </button>
        </div>
      </div>
    )
  }

  if (!videoData) {
    return (
      <div className="timeline-page">
        <div className="error-message">
          <h3>Video Not Found</h3>
          <p>The requested video could not be found.</p>
          <button 
            className="btn" 
            onClick={() => navigate('/')}
            style={{ marginTop: '1rem' }}
          >
            <ArrowLeft size={16} />
            Back to Upload
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="timeline-page">
      {/* Header */}
      <div className="timeline-header">
        <div className="flex-between">
          <button 
            className="btn btn-secondary"
            onClick={() => navigate('/')}
          >
            <ArrowLeft size={16} />
            Back to Upload
          </button>
          
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Sparkles size={20} color="#667eea" />
            <span style={{ color: '#667eea', fontWeight: '600' }}>AI Powered</span>
          </div>
        </div>
        
        <h1>
          <Video size={32} />
          {videoData.filename}
        </h1>
        
        <div className="video-info">
          <div><strong>Video ID:</strong> {videoData.video_id}</div>
          <div><strong>Total Scenes:</strong> {videoData.total_scenes}</div>
          <div><strong>Total Duration:</strong> {formatTime(videoData.statistics?.total_duration || 0)}</div>
          <div><strong>Processed:</strong> {new Date(videoData.processed_at).toLocaleString()}</div>
        </div>

        {/* Enhanced Search Bar */}
        <div className="search-container">
          <Search size={20} style={{ position: 'absolute', left: '25px', top: '50%', transform: 'translateY(-50%)', color: '#a0aec0' }} />
          <input
            type="text"
            className="search-input"
            placeholder="Search scenes by tags or type (animation, outdoor, action, dialogue, etc.)..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
          {searchQuery && filteredScenes.length > 0 && (
            <div className="search-match-count">
              {filteredScenes.length} match{filteredScenes.length !== 1 ? 'es' : ''}
            </div>
          )}
        </div>
      </div>

      {/* Main Content Grid */}
      <div className="main-content">
        
        {/* Left Column - Main Video & Scenes */}
        <div>
          {/* Video Player */}
          <div className="video-player-container">
            <video
              ref={videoRef}
              className="video-player"
              controls
              onPlay={() => setIsPlaying(true)}
              onPause={() => setIsPlaying(false)}
              src={`/api/video/${videoId}`}
            >
              Your browser does not support the video tag.
            </video>
            
            <div style={{ display: 'flex', gap: '1rem', alignItems: 'center', flexWrap: 'wrap' }}>
              <button className="btn" onClick={handlePlayPause}>
                {isPlaying ? <Pause size={16} /> : <Play size={16} />}
                {isPlaying ? 'Pause' : 'Play'}
              </button>
              <div style={{ color: '#718096', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Clock size={16} />
                Current: {formatTime(currentTime)}
              </div>
              {activeScene && (
                <div style={{ 
                  background: '#f0f4ff', 
                  padding: '8px 12px', 
                  borderRadius: '20px',
                  fontSize: '0.9rem',
                  color: '#667eea',
                  fontWeight: '500'
                }}>
                  🎬 {activeScene.scene_class}
                </div>
              )}
            </div>
          </div>

          {/* Active Scene Info */}
          {activeScene && (
            <div className="video-player-container">
              <h3 style={{ marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Tag size={20} />
                Current Scene: {activeScene.scene_id}
                {activeScene === filteredScenes[0] && searchQuery && (
                  <span className="badge">
                    🔍 First Match
                  </span>
                )}
              </h3>
              <div style={{ 
                display: 'grid', 
                gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', 
                gap: '1rem',
                fontSize: '0.95rem'
              }}>
                <div>
                  <strong>Type:</strong> 
                  <span style={{ 
                    textTransform: 'capitalize',
                    marginLeft: '0.5rem',
                    padding: '2px 8px',
                    background: '#f0f4ff',
                    borderRadius: '12px',
                    fontSize: '0.8rem'
                  }}>
                    {activeScene.scene_class}
                  </span>
                </div>
                <div>
                  <strong>Confidence:</strong> {(activeScene.confidence * 100).toFixed(1)}%
                </div>
                <div>
                  <strong>Time:</strong> {formatTime(activeScene.start_time)} - {formatTime(activeScene.end_time)}
                </div>
                <div>
                  <strong>Duration:</strong> {formatTime(activeScene.end_time - activeScene.start_time)}
                </div>
              </div>
              
              <div style={{ marginTop: '1rem' }}>
                <strong>Tags:</strong>
                <div className="scene-tags" style={{ marginTop: '0.5rem' }}>
                  {activeScene.tags.map((tag, tagIndex) => (
                    <span 
                      key={tagIndex} 
                      className={`tag ${tag.toLowerCase().includes(searchQuery.toLowerCase()) ? 'tag-highlight' : ''}`}
                      onClick={() => handleTagClick(tag)}
                      style={{ cursor: 'pointer', transition: 'all 0.2s' }}
                      title="Click to filter scenes with this tag"
                    >
                      {tag}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Scenes Grid */}
          <div className="scenes-section">
            <h2>
              <Sparkles size={24} />
              Detected Scenes 
              {searchQuery && (
                <span style={{ fontSize: '1rem', color: '#718096', marginLeft: '1rem' }}>
                  ({filteredScenes.length} match{filteredScenes.length !== 1 ? 'es' : ''} for "{searchQuery}")
                </span>
              )}
            </h2>
            
            <div className="scenes-grid">
              {filteredScenes.map((scene, index) => (
                <div
                  key={scene.scene_id}
                  className={`scene-card ${activeScene?.scene_id === scene.scene_id ? 'active' : ''} ${index === 0 && searchQuery ? 'search-match' : ''} scene-${scene.scene_class}`}
                  onClick={() => seekToTime(scene.start_time)}
                >
                  <div className="scene-header">
                    <div className="scene-title">
                      {scene.scene_id}
                      {index === 0 && searchQuery && (
                        <span className="badge badge-warning">
                          First Match
                        </span>
                      )}
                    </div>
                    <div className="scene-confidence">
                      {(scene.confidence * 100).toFixed(0)}%
                    </div>
                  </div>
                  
                  <div className="scene-time">
                    ⏱️ {formatTime(scene.start_time)} - {formatTime(scene.end_time)}
                    <span style={{ marginLeft: 'auto', fontSize: '0.8rem', color: '#a0aec0' }}>
                      ({formatTime(scene.end_time - scene.start_time)})
                    </span>
                  </div>
                  
                  <div className="scene-tags">
                    {scene.tags.slice(0, 6).map((tag, tagIndex) => (
                      <span 
                        key={tagIndex} 
                        className={`tag ${tag.toLowerCase().includes(searchQuery.toLowerCase()) ? 'tag-highlight' : ''}`}
                        onClick={(e) => {
                          e.stopPropagation()
                          handleTagClick(tag)
                        }}
                        style={{ cursor: 'pointer', transition: 'all 0.2s' }}
                        title="Click to filter scenes with this tag"
                      >
                        {tag}
                      </span>
                    ))}
                    {scene.tags.length > 6 && (
                      <span className="tag">+{scene.tags.length - 6}</span>
                    )}
                  </div>
                  
                  <div className="scene-meta">
                    <span>🎨 {scene.scene_class}</span>
                    <span>•</span>
                    <span>🖼️ {scene.thumbnail_path ? 'Thumbnail' : 'No thumbnail'}</span>
                    <span>•</span>
                    <span>🎥 {scene.clip_path ? 'Clip ready' : 'Processing clip'}</span>
                  </div>
                </div>
              ))}
            </div>

            {filteredScenes.length === 0 && searchQuery && (
              <div style={{ 
                textAlign: 'center', 
                padding: '4rem 2rem', 
                color: '#718096',
                background: 'white',
                borderRadius: '16px',
                boxShadow: '0 4px 20px rgba(0, 0, 0, 0.08)'
              }}>
                <Search size={48} color="#a0aec0" style={{ marginBottom: '1rem' }} />
                <h3 style={{ color: '#4a5568', marginBottom: '0.5rem' }}>No scenes found</h3>
                <p>No scenes match "{searchQuery}". Try searching for different tags like "animation", "outdoor", "action", etc.</p>
              </div>
            )}
          </div>
        </div>

        {/* RIGHT SIDEBAR - Scene Clips & Analytics */}
        <div className="sidebar">
          {/* Scene Clips Section */}
          <div className="sidebar-section">
            <h3>
              <Video size={20} />
              Scene Clips
              <span style={{ 
                background: '#667eea', 
                color: 'white', 
                padding: '4px 10px', 
                borderRadius: '12px', 
                fontSize: '0.8rem',
                marginLeft: 'auto'
              }}>
                {videoData.scenes?.length || 0}
              </span>
            </h3>

            {/* Currently Playing Clip */}
            {playingClip && (
              <div className="clip-player">
                <h4>🎥 Now Playing: {playingClip.scene.scene_id}</h4>
                <video
                  controls
                  autoPlay
                  style={{ width: '100%', borderRadius: '12px', marginBottom: '1rem' }}
                  onEnded={stopClip}
                  src={playingClip.clipUrl}
                >
                  Your browser does not support the video tag.
                </video>
                <button className="btn btn-secondary" onClick={stopClip} style={{ width: '100%' }}>
                  Stop Clip
                </button>
              </div>
            )}

            {/* Scene Clips Grid */}
            <div className="clips-grid">
              {videoData.scenes?.map((scene) => (
                <div key={scene.scene_id} className="clip-card">
                  <div className="clip-header">
                    <div className="clip-title">{scene.scene_id}</div>
                    <div className="scene-confidence" style={{ fontSize: '0.75rem' }}>
                      {(scene.confidence * 100).toFixed(0)}%
                    </div>
                  </div>
                  
                  <div className="clip-preview">
                    {scene.thumbnail_path ? (
                      <img 
                        src={`/api/thumbnails/${videoId}/${scene.scene_id}.jpg`}
                        alt={scene.scene_id}
                        style={{ width: '100%', height: '100px', objectFit: 'cover' }}
                        onClick={() => playClip(scene)}
                      />
                    ) : scene.clip_path ? (
                      <video
                        style={{ width: '100%', height: '100px', objectFit: 'cover' }}
                        onClick={() => playClip(scene)}
                      >
                        <source src={`/api/clips/${videoId}/${scene.scene_id}.mp4`} type="video/mp4" />
                      </video>
                    ) : (
                      <div style={{ 
                        background: '#f0f0f0', 
                        borderRadius: '8px', 
                        height: '100px', 
                        display: 'flex', 
                        alignItems: 'center', 
                        justifyContent: 'center',
                        color: '#666',
                        fontSize: '0.9rem'
                      }}>
                        Clip Processing...
                      </div>
                    )}
                  </div>
                  
                  <div className="clip-info">
                    <div style={{ fontSize: '0.8rem', color: '#718096', marginBottom: '0.5rem' }}>
                      ⏱️ {formatTime(scene.start_time)} - {formatTime(scene.end_time)}
                    </div>
                    
                    <div className="scene-tags">
                      {scene.tags.slice(0, 3).map((tag, tagIndex) => (
                        <span 
                          key={tagIndex} 
                          className={`tag ${tag.toLowerCase().includes(searchQuery.toLowerCase()) ? 'tag-highlight' : ''}`}
                          style={{ fontSize: '0.7rem', cursor: 'pointer' }}
                          onClick={(e) => {
                            e.stopPropagation()
                            handleTagClick(tag)
                          }}
                          title="Click to filter scenes"
                        >
                          {tag}
                        </span>
                      ))}
                    </div>
                    
                    <div className="clip-actions">
                      <button 
                        className="btn"
                        style={{ 
                          padding: '8px 12px',
                          fontSize: '0.8rem'
                        }}
                        onClick={() => seekToTime(scene.start_time)}
                      >
                        <Play size={12} />
                        Jump to Scene
                      </button>
                      
                      {scene.clip_path && (
                        <button 
                          className="btn btn-secondary"
                          style={{ 
                            padding: '8px 12px',
                            fontSize: '0.8rem'
                          }}
                          onClick={() => playClip(scene)}
                        >
                          <Video size={12} />
                          Play Clip
                        </button>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Classification Summary */}
          <div className="sidebar-section">
            <h3>
              <BarChart3 size={20} />
              Scene Analytics
            </h3>
            <div className="classification-stats">
              {videoData.statistics?.scene_types && Object.entries(videoData.statistics.scene_types).map(([sceneType, count]) => {
                const percentage = ((count / videoData.total_scenes) * 100).toFixed(1)
                
                return (
                  <div key={sceneType} className="stat-item">
                    <div className="stat-label">{sceneType.replace('_', ' ')}</div>
                    <div className="stat-bar">
                      <div 
                        className="stat-fill" 
                        style={{ width: `${percentage}%` }}
                      ></div>
                    </div>
                    <div className="stat-value">{count} ({percentage}%)</div>
                  </div>
                )
              })}
              
              {videoData.statistics?.average_confidence && (
                <div className="stat-item">
                  <div className="stat-label">Avg Confidence</div>
                  <div className="stat-bar">
                    <div 
                      className="stat-fill" 
                      style={{ width: `${(videoData.statistics.average_confidence * 100).toFixed(1)}%` }}
                    ></div>
                  </div>
                  <div className="stat-value">{(videoData.statistics.average_confidence * 100).toFixed(1)}%</div>
                </div>
              )}
            </div>
          </div>

          {/* Popular Tags */}
          <div className="sidebar-section">
            <h3>
              <Tag size={20} />
              Popular Tags
            </h3>
            <div className="scene-tags">
              {videoData.statistics?.tags_summary && 
                Object.entries(videoData.statistics.tags_summary)
                  .sort(([,a], [,b]) => b - a)
                  .slice(0, 12)
                  .map(([tag, count]) => (
                    <span 
                      key={tag} 
                      className="tag"
                      style={{ fontSize: '0.75rem' }}
                      onClick={() => setSearchQuery(tag)}
                    >
                      {tag} ({count})
                    </span>
                  ))
              }
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default TimelinePage