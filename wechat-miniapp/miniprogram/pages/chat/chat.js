const api = require('../../utils/api')
const files = require('../../utils/files')

const EMOJIS = ['😀','😃','😄','😁','😊','🙂','🙃','😉','🥰','😍','😘','😋','😎','🤩','🥳','😂','🤣','🥲','🥹','😅','😌','😏','😒','🙄','😔','😞','😢','😭','😤','😠','😡','🤬','😳','🥺','😱','😨','🤔','🫠','🫡','🤭','🤫','😴','🤒','🤧','🤡','👻','💀','👀','👍','👎','👏','🙌','🙏','🤝','👌','✌️','🤞','💪','🫶','❤️','💕','💔','🔥','✨','🎉','💯']

function localDay(value) {
  const d = new Date(value)
  return `${d.getFullYear()}-${d.getMonth() + 1}-${d.getDate()}`
}

function timeLabel(value) {
  const d = new Date(value), now = new Date()
  const clock = `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate())
  const day = new Date(d.getFullYear(), d.getMonth(), d.getDate())
  const days = Math.floor((today - day) / 86400000)
  if (days === 0) return clock
  if (days > 0 && days < 7) return '星期' + '日一二三四五六'[d.getDay()] + ' ' + clock
  return `${d.getFullYear()}年${d.getMonth() + 1}月${d.getDate()}日 ${clock}`
}

Page({
  data: {
    authenticated: !!getApp().globalData.token,
    tokenInput: '',
    ready: false,
    error: '',
    showConsent: false,
    messages: [],
    sessionId: '',
    draft: '',
    selectedAsset: '',
    selectedAssetPath: '',
    uploading: false,
    busy: false,
    waitingForUser: false,
    scrollInto: 'bottom',
    showEmoji: false,
    emojis: EMOJIS,
    showLibrary: false,
    library: [],
    libraryImporting: false,
    libraryProgress: '',
    showSurvey: false,
    surveyWaves: [
      { label: '首次试聊后', value: 'baseline' },
      { label: '第 7 天', value: 'day7' },
      { label: '第 14 天', value: 'day14' }
    ],
    surveyWaveIndex: 0,
    surveyAnswers: [4, 4, 4]
  },

  onLoad(options) {
    let incoming = options.token || ''
    if (!incoming && options.scene) incoming = decodeURIComponent(options.scene)
    if (incoming) {
      api.saveToken(api.tokenFromText(incoming))
      this.setData({ authenticated: true })
    }
    this.turnSeq = 0
    if (this.data.authenticated) this.initialize()
  },

  onShow() {
    if (this.didLoad && this.data.authenticated && !this.data.busy) this.refreshState()
    this.didLoad = true
  },

  onUnload() {
    clearTimeout(this.pauseTimer)
  },

  onTokenInput(e) { this.setData({ tokenInput: e.detail.value }) },

  loginWithToken() {
    const token = api.tokenFromText(this.data.tokenInput)
    if (!token) return this.setData({ error: '请输入个人链接或凭证' })
    api.saveToken(token)
    this.setData({ authenticated: true, error: '' })
    this.initialize()
  },

  scanParticipantLink() {
    wx.scanCode({
      onlyFromCamera: false,
      success: result => {
        const token = api.tokenFromText(result.result)
        if (!token) return this.setData({ error: '二维码中没有个人凭证' })
        api.saveToken(token)
        this.setData({ authenticated: true, error: '' })
        this.initialize()
      },
      fail: e => { if (!String(e.errMsg).includes('cancel')) this.setData({ error: '没有识别到个人链接' }) }
    })
  },

  async initialize() {
    wx.showNavigationBarLoading()
    try {
      const health = await api.request('/api/health')
      this.setData({ ready: !!health.ready })
      await this.refreshState()
    } catch (e) {
      if (e.status === 403) {
        api.saveToken('')
        this.setData({ authenticated: false, error: '个人凭证无效，请重新扫码进入' })
      } else this.setData({ error: e.message || '连接失败' })
    } finally { wx.hideNavigationBarLoading() }
  },

  async refreshState() {
    const state = await api.request('/api/state')
    if (!state.consent) {
      this.setData({ showConsent: true })
      return
    }
    let latest = state.sessions[0]
    if (!latest || localDay(latest.created) !== localDay(Date.now())) latest = await api.request('/api/session', {})
    const messages = []
    let previous = 0
    for (const row of state.messages) {
      const created = new Date(row.created).getTime()
      const parts = row.role === 'assistant' ? String(row.text || '').split(/\n+/).filter(Boolean) : [row.text || '']
      parts.forEach((text, partIndex) => messages.push({
        key: row.id + '-' + partIndex,
        id: row.id,
        role: row.role,
        text,
        asset: partIndex === parts.length - 1 ? row.asset : '',
        assetPath: '',
        showTime: partIndex === 0 && (!previous || created - previous >= 300000),
        time: timeLabel(row.created)
      }))
      previous = created
    }
    this.setData({ messages, sessionId: latest.id, showConsent: false, scrollInto: 'bottom' })
    messages.forEach((item, index) => {
      if (item.asset) files.assetPath(item.asset).then(path => this.setData({ [`messages[${index}].assetPath`]: path })).catch(e => this.setData({ error: e.message }))
    })
  },

  async agreeConsent() {
    try { await api.request('/api/consent', {}); await this.refreshState() }
    catch (e) { this.setData({ error: e.message }) }
  },

  onDraftInput(e) {
    this.setData({ draft: e.detail.value })
    if (!this.data.busy) return
    clearTimeout(this.pauseTimer)
    this.setData({ waitingForUser: true })
    this.pauseTimer = setTimeout(() => this.setData({ waitingForUser: false }), 2000)
  },

  async sendMessage() {
    const text = this.data.draft.trim(), asset = this.data.selectedAsset
    if (!text && !asset) return
    if (this.data.busy) {
      const oldTurn = this.activeTurn
      this.turnSeq += 1
      clearTimeout(this.pauseTimer)
      try { await api.request('/api/cancel', { turn: oldTurn }) } catch (_) {}
    }
    const seq = ++this.turnSeq
    const turn = `${Date.now()}-${Math.random().toString(16).slice(2)}`
    this.activeTurn = turn
    const optimistic = this.data.messages.slice()
    if (text) optimistic.push({ key: turn + '-text', role: 'user', text, showTime: false })
    if (asset) optimistic.push({ key: turn + '-image', role: 'user', assetPath: this.data.selectedAssetPath, showTime: false })
    optimistic.push({ key: turn + '-typing', role: 'assistant', typing: true, showTime: false })
    this.setData({ messages: optimistic, draft: '', selectedAsset: '', selectedAssetPath: '', busy: true, waitingForUser: false, error: '', scrollInto: 'bottom' })
    try {
      await api.request('/api/chat', { session: this.data.sessionId, text, asset, turn })
      if (seq !== this.turnSeq) return
      await this.refreshState()
    } catch (e) {
      if (seq !== this.turnSeq || (e.data && e.data.interrupted)) return
      this.setData({ draft: text, selectedAsset: asset, error: e.message || '发送失败' })
      if (asset) files.assetPath(asset).then(path => this.setData({ selectedAssetPath: path }))
      await this.refreshState().catch(() => {})
    } finally {
      if (seq === this.turnSeq) this.setData({ busy: false, waitingForUser: false })
    }
  },

  async importPaths(paths, selectFirst = false) {
    let ok = 0, failed = 0, first = null
    this.setData({ libraryImporting: true, libraryProgress: `正在导入 0/${paths.length}` })
    for (const path of paths) {
      try {
        const image = await files.readBase64(path)
        const item = await api.request('/api/library/import', { image })
        if (!first) first = { id: item.id, path }
        ok += 1
      } catch (e) { failed += 1; this.setData({ error: e.message || '导入失败' }) }
      this.setData({ libraryProgress: `正在导入 ${ok + failed}/${paths.length}` })
    }
    this.setData({ libraryImporting: false, libraryProgress: failed ? `已导入 ${ok} 张，${failed} 张失败` : `已导入 ${ok} 张` })
    if (selectFirst && first) this.setData({ selectedAsset: first.id, selectedAssetPath: first.path, showLibrary: false })
    else await this.loadLibrary()
  },

  chooseOneImage() {
    wx.chooseMedia({ count: 1, mediaType: ['image'], sourceType: ['album', 'camera'], success: r => this.importPaths(r.tempFiles.map(x => x.tempFilePath), true) })
  },

  importFromAlbum() {
    wx.chooseMedia({ count: 9, mediaType: ['image'], sourceType: ['album'], success: r => this.importPaths(r.tempFiles.map(x => x.tempFilePath)) })
  },

  importFromWechat() {
    wx.chooseMessageFile({ count: 100, type: 'image', success: r => this.importPaths(r.tempFiles.map(x => x.path)) })
  },

  async openLibrary() {
    this.setData({ showLibrary: true, showEmoji: false, libraryProgress: '' })
    await this.loadLibrary().catch(e => this.setData({ error: e.message }))
  },

  async loadLibrary() {
    const result = await api.request('/api/library')
    const library = result.items.map(x => ({ ...x, path: '' }))
    this.setData({ library })
    library.forEach((item, index) => files.assetPath(item.id).then(path => this.setData({ [`library[${index}].path`]: path })).catch(() => {}))
  },

  selectLibraryAsset(e) {
    const { id, path } = e.currentTarget.dataset
    if (!path) return
    this.setData({ selectedAsset: id, selectedAssetPath: path, showLibrary: false })
  },

  removeSelectedAsset() { this.setData({ selectedAsset: '', selectedAssetPath: '', uploading: false }) },
  toggleEmoji() { this.setData({ showEmoji: !this.data.showEmoji, showLibrary: false }) },
  pickEmoji(e) { this.setData({ draft: this.data.draft + e.currentTarget.dataset.emoji, showEmoji: false }) },
  closeSheets() { this.setData({ showEmoji: false, showLibrary: false }) },
  stopTap() {},
  previewMessageImage(e) { wx.previewImage({ urls: [e.currentTarget.dataset.src] }) },

  openSurvey() { this.setData({ showSurvey: true }) },
  closeSurvey() { this.setData({ showSurvey: false }) },
  changeSurveyWave(e) { this.setData({ surveyWaveIndex: Number(e.detail.value) }) },
  surveyOne(e) { this.setData({ 'surveyAnswers[0]': e.detail.value }) },
  surveyTwo(e) { this.setData({ 'surveyAnswers[1]': e.detail.value }) },
  surveyThree(e) { this.setData({ 'surveyAnswers[2]': e.detail.value }) },
  async submitSurvey() {
    try {
      const wave = this.data.surveyWaves[this.data.surveyWaveIndex].value
      await api.request('/api/survey', { wave, answers: this.data.surveyAnswers.map(Number) })
      this.setData({ showSurvey: false })
      wx.showToast({ title: '已提交', icon: 'success' })
    } catch (e) { this.setData({ error: e.message }) }
  }
})
