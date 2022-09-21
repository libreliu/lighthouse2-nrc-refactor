/* main_tools.h - Copyright 2019/2021 Utrecht University

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

	   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

GLFWwindow* window = 0;

void disableAllAuxRT() {
	for (auto &rt: auxRenderTargets) {
		renderer->DisableAuxTargetExt(rt.rtName.c_str());
	}
	auxRenderTargets.clear();
}

//  +-----------------------------------------------------------------------------+
//  |  ...Callback                                                                |
//  |  Various GLFW callbacks, mostly just forwarded to AntTweakBar.        LH2'19|
//  +-----------------------------------------------------------------------------+
void ReshapeWindowCallback( GLFWwindow* window, int w, int h )
{
	// don't resize if nothing changed or the window was minimized
	if ((scrwidth == w && scrheight == h) || w == 0 || h == 0) return;
	scrwidth = w, scrheight = h;
	delete renderTarget;
	renderTarget = new GLTexture( scrwidth, scrheight, GLTexture::FLOAT );
	glViewport( 0, 0, scrwidth, scrheight );
	renderer->SetTarget( renderTarget, 1 );

	if (!auxRTEnabled) return;
	disableAllAuxRT();

	// name1;name2;name3;...;lastname;
	std::string auxRTSemiColonList = renderer->GetSettingStringExt("auxiliaryRenderTargets");
	
	std::vector<std::string> rtNames;
    size_t start = 0U;
    auto end = auxRTSemiColonList.find(";");
    while (end != std::string::npos) {
        rtNames.push_back(auxRTSemiColonList.substr(start, end - start));
        start = end + 1;
        end = auxRTSemiColonList.find(";", start);
    }

	for (auto &rtName: rtNames) {
		auxRenderTargets.push_back(RenderTarget{
			rtName,
			std::make_unique<GLTexture>(scrwidth, scrheight, GLTexture::FLOAT),
			false
		});
		renderer->EnableAuxTargetExt(
			auxRenderTargets.back().rtName.c_str(),
			auxRenderTargets.back().texture.get()
		);
		renderer->SettingStringExt("clearAuxTargetAccumulative", auxRenderTargets.back().rtName.c_str());
	}
}

void KeyEventCallback( GLFWwindow* window, int key, int scancode, int action, int mods )
{
	if (key == GLFW_KEY_ESCAPE) running = false;
	if (action == GLFW_PRESS) keystates[key] = true;
	else if (action == GLFW_RELEASE) keystates[key] = false;
}
void CharEventCallback( GLFWwindow* window, uint code ) { /* nothing here yet */ }
void MouseButtonCallback( GLFWwindow* window, int button, int action, int mods ) { /* nothing here yet */ }
void MousePosCallback( GLFWwindow* window, double x, double y )
{
	// set pixel probe pos for triangle picking
	if (renderer) renderer->SetProbePos( make_int2( (int)x, (int)y ) );
}
void ErrorCallback( int error, const char*description )
{
	fprintf( stderr, "GLFW Error: %s\n", description );
}

//  +-----------------------------------------------------------------------------+
//  |  InitGLFW                                                                   |
//  |  Opens a GL window using GLFW.                                        LH2'19|
//  +-----------------------------------------------------------------------------+
void OpenConsole();
void InitGLFW()
{
	// open a window
	if (!glfwInit()) exit( EXIT_FAILURE );
	glfwSetErrorCallback( ErrorCallback );
	glfwWindowHint( GLFW_CONTEXT_VERSION_MAJOR, 4 );
	glfwWindowHint( GLFW_CONTEXT_VERSION_MINOR, 5 );
	glfwWindowHint( GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE );
	glfwWindowHint( GLFW_RESIZABLE, GL_TRUE );
	// DELETED: Use single buffering
	// glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_FALSE);
	if (!(window = glfwCreateWindow( SCRWIDTH, SCRHEIGHT, "LightHouse v2.0", nullptr, nullptr ))) exit( EXIT_FAILURE );
	glfwMakeContextCurrent( window );
	// register callbacks
	glfwSetFramebufferSizeCallback( window, ReshapeWindowCallback );
	glfwSetKeyCallback( window, KeyEventCallback );
	glfwSetMouseButtonCallback( window, MouseButtonCallback );
	glfwSetCursorPosCallback( window, MousePosCallback );
	glfwSetCharCallback( window, CharEventCallback );
	// initialize GLAD
	if (!gladLoadGLLoader( (GLADloadproc)glfwGetProcAddress )) exit( EXIT_FAILURE );
	// prepare OpenGL state
	glDisable( GL_DEPTH_TEST );
	glDisable( GL_CULL_FACE );
	glDisable( GL_BLEND );
	// logo
	GLTexture* logo = new GLTexture( "data/system/logo.png", GL_LINEAR );
	shader = new Shader( "shaders/vignette.vert", "shaders/vignette.frag" );
	shader->Bind();
	shader->SetInputTexture( 0, "color", logo );
	float hscale = ((float)SCRHEIGHT / SCRWIDTH) * ((float)logo->width / logo->height);
	shader->SetInputMatrix( "view", mat4::Scale( make_float3( 0.1f * hscale, 0.1f, 1 ) ) );
	DrawQuad();
	shader->Unbind();
	// DELETED: Use single buffering instead of double buffering
	glfwSwapBuffers( window );
	// glFinish();
	delete logo;
	// we want a console window for text output
	OpenConsole();
}

//  +-----------------------------------------------------------------------------+
//  |  OpenConsole                                                                |
//  |  Create the console window for text output.                           LH2'19|
//  +-----------------------------------------------------------------------------+
void OpenConsole()
{
#ifdef _MSC_VER
	CONSOLE_SCREEN_BUFFER_INFO coninfo;
	AllocConsole();
	GetConsoleScreenBufferInfo( GetStdHandle( STD_OUTPUT_HANDLE ), &coninfo );
	coninfo.dwSize.X = 1280;
	coninfo.dwSize.Y = 800;
	SetConsoleScreenBufferSize( GetStdHandle( STD_OUTPUT_HANDLE ), coninfo.dwSize );
	FILE* file = nullptr;
	freopen_s( &file, "CON", "w", stdout );
	freopen_s( &file, "CON", "w", stderr );
	SetWindowPos( GetConsoleWindow(), HWND_TOP, 0, 0, 1280, 800, 0 );
	glfwShowWindow( window );
#endif
}

// EOF