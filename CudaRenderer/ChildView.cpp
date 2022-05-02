
// ChildView.cpp : implementation of the CChildView class
//

#include "pch.h"
#include "framework.h"
#include "CudaRenderer.h"
#include "ChildView.h"
#include "renderer.h"
#include "MainFrm.h"

#include <string>
#include <iostream>
#include <sstream>
#include <iomanip>

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

// CChildView

CChildView::CChildView()
{
}

CChildView::~CChildView()
{
}


BEGIN_MESSAGE_MAP(CChildView, COpenGLWnd)
	ON_WM_PAINT()
	ON_WM_KEYDOWN()
	ON_WM_KEYUP()
END_MESSAGE_MAP()



// CChildView message handlers

BOOL CChildView::PreCreateWindow(CREATESTRUCT& cs) 
{
	if (!COpenGLWnd::PreCreateWindow(cs))
		return FALSE;

	cs.dwExStyle |= WS_EX_CLIENTEDGE;
	cs.style &= ~WS_BORDER;
	cs.lpszClass = AfxRegisterWndClass(CS_HREDRAW|CS_VREDRAW|CS_DBLCLKS, 
		::LoadCursor(nullptr, IDC_ARROW), reinterpret_cast<HBRUSH>(COLOR_WINDOW+1), nullptr);

	return TRUE;
}

void CChildView::OnGLDraw(CDC* pDC)
{
	int width, height;
	GetSize(width, height);
	renderer.nx = width;
	renderer.ny = height;
	if (!initialized) {

		initialized = true;
		start = 0;
		renderer.Render_Init();
	}

	renderer.render();
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, width, 0, height, -1, 1);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	
	glRasterPos3i(0, 0, 0);
	glDrawPixels(renderer.nx, renderer.ny,
		GL_RGB, GL_FLOAT, renderer.fb);
	glFlush();

	// fps calculation

	const auto now = std::chrono::high_resolution_clock::now();
	const std::chrono::duration<double, std::milli> diff = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_call);
	last_call = now;

	const double fps = 1000 / diff.count();

	// write FPSs to window title
	std::wstringstream ss;
	ss << std::setprecision(3);
	ss << L"Cuda Renderer (";
	ss << fps;
	ss << L" fps)";

	CMainFrame* pFrame = (CMainFrame*)GetParent();
	pFrame->SetAppName(ss.str().c_str());
	pFrame->OnUpdateFrameTitle(TRUE);
	
	Invalidate();

}

void CChildView::OnKeyDown(UINT nChar, UINT nRepCnt, UINT nFlags)
{

	switch (nChar) {
	case VK_LEFT:
		renderer.keys.left = true;
		break;
	case VK_RIGHT:
		renderer.keys.right = true;
		break;
	case VK_UP:
		renderer.keys.up = true;
		break;
	case VK_DOWN:
		renderer.keys.down = true;
		break;

	case VK_SPACE:
		renderer.keys.space = true;
		break;
	case VK_SHIFT:
		renderer.keys.shift = true;
		break;

	case 0x57:
		renderer.keys.w = true;
		break;

	case 0x41:
		renderer.keys.a = true;
		break;

	case 0x53:
		renderer.keys.s = true;
		break;

	case 0x44:
		renderer.keys.d = true;
		break;

	}

	COpenGLWnd::OnKeyDown(nChar, nRepCnt, nFlags);
}


void CChildView::OnKeyUp(UINT nChar, UINT nRepCnt, UINT nFlags)
{
	switch (nChar) {
	case VK_LEFT:
		renderer.keys.left = false;
		break;
	case VK_RIGHT:
		renderer.keys.right = false;
		break;
	case VK_UP:
		renderer.keys.up = false;
		break;
	case VK_DOWN:
		renderer.keys.down = false;
		break;

	case VK_SPACE:
		renderer.keys.space = false;
		break;
	case VK_SHIFT:
		renderer.keys.shift = false;
		break;

	case 0x57:
		renderer.keys.w = false;
		break;

	case 0x41:
		renderer.keys.a = false;
		break;

	case 0x53:
		renderer.keys.s = false;
		break;

	case 0x44:
		renderer.keys.d = false;
		break;
	}
	COpenGLWnd::OnKeyUp(nChar, nRepCnt, nFlags);
}
