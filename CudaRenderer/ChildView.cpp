
// ChildView.cpp : implementation of the CChildView class
//

#include "pch.h"
#include "framework.h"
#include "CudaRenderer.h"
#include "ChildView.h"
#include "renderer.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif
#include <iostream>


// CChildView

CChildView::CChildView()
{
}

CChildView::~CChildView()
{
}


BEGIN_MESSAGE_MAP(CChildView, COpenGLWnd)
	ON_WM_PAINT()
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
	if (!initialized) {
		renderer.nx = width;
		renderer.ny = height;
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
	Invalidate();

}
