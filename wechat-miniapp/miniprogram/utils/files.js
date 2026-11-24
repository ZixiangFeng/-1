const { request } = require('./api')
const cache = new Map()

function readBase64(path) {
  return new Promise((resolve, reject) => {
    wx.getFileSystemManager().readFile({ filePath: path, encoding: 'base64', success: r => resolve(r.data), fail: reject })
  })
}

async function assetPath(id) {
  if (cache.has(id)) return cache.get(id)
  const result = await request('/api/asset/' + id)
  const path = `${wx.env.USER_DATA_PATH}/xiaoyu-${id}.jpg`
  await new Promise((resolve, reject) => {
    wx.getFileSystemManager().writeFile({ filePath: path, data: result.image, encoding: 'base64', success: resolve, fail: reject })
  })
  cache.set(id, path)
  return path
}

module.exports = { readBase64, assetPath }
