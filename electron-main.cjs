const { app, BrowserWindow } = require('electron');
const path = require('path');

function createWindow() {
  const win = new BrowserWindow({
    width: 1280,
    height: 800,
    title: "AI DJ Studio",
    backgroundColor: '#050505',
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
    },
  });

  // En desarrollo local (si corres npm run dev), Electron cargaría esto:
  // win.loadURL('http://localhost:3000');
  
  // En producción (después de npm run build), cargaría el archivo compilado:
  win.loadFile(path.join(__dirname, 'dist/index.html'));

  // Abre las herramientas de desarrollo si lo deseas
  // win.webContents.openDevTools();
}

app.whenReady().then(() => {
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});
