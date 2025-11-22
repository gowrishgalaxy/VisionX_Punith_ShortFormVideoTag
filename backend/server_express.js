const express = require('express');
const cors = require('cors');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');
const { v4: uuid } = require('uuid');

const app = express();
const PORT = 8000;

// Middleware
app.use(cors());
app.use(express.json());

// Upload configuration
const uploadDir = path.join(__dirname, 'uploads');
const processedDir = path.join(__dirname, 'processed');
if (!fs.existsSync(uploadDir)) fs.mkdirSync(uploadDir, { recursive: true });
if (!fs.existsSync(processedDir)) fs.mkdirSync(processedDir, { recursive: true });
if (!fs.existsSync(path.join(processedDir, 'clips'))) fs.mkdirSync(path.join(processedDir, 'clips'), { recursive: true });
if (!fs.existsSync(path.join(processedDir, 'thumbnails'))) fs.mkdirSync(path.join(processedDir, 'thumbnails'), { recursive: true });

const upload = multer({ 
  dest: uploadDir,
  limits: { fileSize: 500 * 1024 * 1024 } // 500MB
});

// In-memory processing results
const processingResults = {};

// Status endpoint
app.get('/', (req, res) => {
  const totalVideos = Object.keys(processingResults).length;
  const completedVideos = Object.values(processingResults).filter(r => r.status === 'completed').length;
  const processingVideos = Object.values(processingResults).filter(r => r.status === 'processing').length;
  let totalScenes = 0;
  for (const result of Object.values(processingResults)) {
    if (result.status === 'completed' && result.scenes) {
      totalScenes += result.scenes.length;
    }
  }
  
  res.json({
    system_status: 'operational',
    ai_model_loaded: true,
    total_videos_processed: totalVideos,
    completed_videos: completedVideos,
    processing_videos: processingVideos,
    total_scenes_detected: totalScenes,
    timestamp: new Date().toISOString()
  });
});

// Upload endpoint
app.post('/api/upload', upload.single('video'), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No video file provided' });
  }

  const videoId = uuid();
  const originalName = req.file.originalname || 'video.mp4';
  const ext = path.extname(originalName);
  const destPath = path.join(uploadDir, `${videoId}${ext}`);

  // Rename uploaded file
  fs.renameSync(req.file.path, destPath);

  // Initialize processing status
  processingResults[videoId] = {
    status: 'processing',
    video_id: videoId,
    filename: originalName,
    upload_time: new Date().toISOString(),
    progress: 0,
    scenes: [],
    tags: [],
    error: null
  };

  // Start async processing using Python
  processVideoAsync(videoId, destPath);

  res.json({
    success: true,
    video_id: videoId,
    message: 'Video uploaded and processing started'
  });
});

// Status endpoint
app.get('/api/status/:videoId', (req, res) => {
  const { videoId } = req.params;
  const result = processingResults[videoId];

  if (!result) {
    return res.status(404).json({ error: 'Video not found' });
  }

  res.json(result);
});

// Scenes endpoint
app.get('/api/scenes/:videoId', (req, res) => {
  const { videoId } = req.params;
  const result = processingResults[videoId];

  if (!result) {
    return res.status(404).json({ error: 'Video not found' });
  }

  res.json({
    video_id: videoId,
    scenes: result.scenes || [],
    total_scenes: (result.scenes || []).length
  });
});

// Search endpoint
app.get('/api/search/:videoId', (req, res) => {
  const { videoId } = req.params;
  const { query } = req.query;
  const result = processingResults[videoId];

  if (!result) {
    return res.status(404).json({ error: 'Video not found' });
  }

  let filtered = result.scenes || [];
  if (query) {
    filtered = filtered.filter(scene => 
      (scene.tags || []).some(tag => tag.toLowerCase().includes(query.toLowerCase()))
    );
  }

  res.json({
    video_id: videoId,
    query: query,
    scenes: filtered,
    total_scenes: filtered.length
  });
});

// Statistics endpoint
app.get('/api/statistics', (req, res) => {
  const totalVideos = Object.keys(processingResults).length;
  const completedVideos = Object.values(processingResults).filter(r => r.status === 'completed').length;
  const processingVideos = Object.values(processingResults).filter(r => r.status === 'processing').length;
  let totalScenes = 0;
  for (const result of Object.values(processingResults)) {
    if (result.status === 'completed' && result.scenes) {
      totalScenes += result.scenes.length;
    }
  }
  
  res.json({
    system_status: 'operational',
    total_videos_processed: totalVideos,
    completed_videos: completedVideos,
    processing_videos: processingVideos,
    total_scenes_detected: totalScenes,
    timestamp: new Date().toISOString()
  });
});

// Clips endpoint (stub)
app.get('/api/clips/:videoId/:sceneId', (req, res) => {
  res.json({ message: 'Clip extraction coming soon' });
});

// Thumbnails endpoint (stub)
app.get('/api/thumbnails/:videoId/:sceneId', (req, res) => {
  res.json({ message: 'Thumbnail generation coming soon' });
});

// Process video asynchronously using Python
function processVideoAsync(videoId, filePath) {
  const pythonScript = path.join(__dirname, 'process_video.py');
  
  const pythonProcess = spawn('python', [pythonScript, filePath, videoId], {
    cwd: __dirname,
    stdio: ['pipe', 'pipe', 'pipe']
  });

  let outputData = '';
  let errorData = '';

  pythonProcess.stdout.on('data', (data) => {
    outputData += data.toString();
    console.log(`[${videoId}] stdout: ${data}`);
  });

  pythonProcess.stderr.on('data', (data) => {
    errorData += data.toString();
    console.log(`[${videoId}] stderr: ${data}`);
  });

  pythonProcess.on('close', (code) => {
    if (code === 0 && outputData) {
      try {
        const result = JSON.parse(outputData);
        processingResults[videoId] = {
          ...result,
          video_id: videoId,
          status: 'completed'
        };
        console.log(`[${videoId}] Processing completed successfully`);
      } catch (e) {
        processingResults[videoId].status = 'error';
        processingResults[videoId].error = 'Failed to parse processing results';
        console.error(`[${videoId}] Parse error: ${e.message}`);
      }
    } else {
      processingResults[videoId].status = 'error';
      processingResults[videoId].error = errorData || 'Processing failed';
      console.error(`[${videoId}] Processing error: ${errorData}`);
    }
  });

  pythonProcess.on('error', (err) => {
    processingResults[videoId].status = 'error';
    processingResults[videoId].error = err.message;
    console.error(`[${videoId}] Process error: ${err.message}`);
  });
}

// Start server
app.listen(PORT, '0.0.0.0', () => {
  console.log(`[+] Express server running on http://0.0.0.0:${PORT}`);
  console.log(`[+] Accessible at http://localhost:${PORT}`);
  console.log('[*] Available Endpoints:');
  console.log('   GET  /                       - API status');
  console.log('   POST /api/upload             - Upload video for AI processing');
  console.log('   GET  /api/status/<video_id>  - Check processing status');
  console.log('   GET  /api/scenes/<video_id>  - Get processed scenes');
  console.log('   GET  /api/search/<video_id>  - Search scenes by tags');
  console.log('   GET  /api/statistics         - System statistics');
  console.log('   GET  /api/clips/<vid>/<sid>  - Get scene clips');
  console.log('   GET  /api/thumbnails/<vid>/<sid> - Get thumbnails');
});

process.on('SIGINT', () => {
  console.log('\n[*] Server shutting down...');
  process.exit(0);
});
