const { API_BASE_URL } = require('../config')

function token() {
  return getApp().globalData.token || wx.getStorageSync('xiaoyuParticipantToken') || ''
}

function request(path, data, method = data ? 'POST' : 'GET') {
  return new Promise((resolve, reject) => {
    const task = wx.request({
      url: API_BASE_URL + path,
      method,
      data,
      timeout: 240000,
      header: {
        'Content-Type': 'application/json',
        Authorization: 'Bearer ' + token()
      },
      success(res) {
        if (res.statusCode >= 200 && res.statusCode < 300) resolve(res.data)
        else reject(Object.assign(new Error((res.data && res.data.error) || '请求失败'), { status: res.statusCode, data: res.data }))
      },
      fail: reject
    })
    if (path === '/api/chat') getApp().globalData.chatTask = task
  })
}

function saveToken(value) {
  const clean = String(value || '').trim()
  getApp().globalData.token = clean
  wx.setStorageSync('xiaoyuParticipantToken', clean)
}

function tokenFromText(value) {
  const text = decodeURIComponent(String(value || '').trim())
  const match = text.match(/(?:#|[?&])token=([^&#]+)/)
  return match ? decodeURIComponent(match[1]) : text
}

module.exports = { API_BASE_URL, request, saveToken, tokenFromText }
