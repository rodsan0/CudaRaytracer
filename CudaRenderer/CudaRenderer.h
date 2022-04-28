
// CudaRenderer.h : main header file for the CudaRenderer application
//
#pragma once

#ifndef __AFXWIN_H__
	#error "include 'pch.h' before including this file for PCH"
#endif

#include "resource.h"       // main symbols


// CCudaRendererApp:
// See CudaRenderer.cpp for the implementation of this class
//

class CCudaRendererApp : public CWinApp
{
public:
	CCudaRendererApp() noexcept;


// Overrides
public:
	virtual BOOL InitInstance();
	virtual int ExitInstance();

// Implementation

public:
	afx_msg void OnAppAbout();
	DECLARE_MESSAGE_MAP()
};

extern CCudaRendererApp theApp;
