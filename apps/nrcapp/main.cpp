/* main.cpp - Copyright 2019/2021 Utrecht University

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

#include "platform.h"
#include "rendersystem.h"
#include <bitset>
#include <memory>
#include <chrono>

// Dear ImGui
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include "implot.h"

const int ScrollingBufferMaxSize = 2000;

struct ScrollingBuffer {
    int MaxSize;
    int Offset;
    ImVector<ImVec2> Data;
    ScrollingBuffer(int max_size = ScrollingBufferMaxSize) {
        MaxSize = max_size;
        Offset  = 0;
        Data.reserve(MaxSize);
    }
    void AddPoint(float x, float y) {
        if (Data.size() < MaxSize)
            Data.push_back(ImVec2(x,y));
        else {
            Data[Offset] = ImVec2(x,y);
            Offset =  (Offset + 1) % MaxSize;
        }
    }
    void Erase() {
        if (Data.size() > 0) {
            Data.shrink(0);
            Offset  = 0;
        }
    }
};


struct RenderTarget {
	std::string rtName;
	std::unique_ptr<GLTexture> texture;
	bool accumulative;
};

std::string currentRT;
bool nrcTrainingEnable = false;
bool auxRTEnabled;
int nrcNumInitialTrainingRays = 1;
enum nrcRenderModeSet {
	ORIGINAL = 0,
	REFERENCE,
	NRC_PRIMARY,
	NRC_FULL
} nrcRenderMode;
int frameRendered = 0;
bool renderConverge = false;
int trainVisLayer = 0;
int nrcMaxTrainPathLength = 0;

static RenderAPI* renderer = 0;
static GLTexture* renderTarget = 0;
static std::vector<RenderTarget> auxRenderTargets;
static Shader* shader = 0;
static uint scrwidth = 0, scrheight = 0, car = 0, scrspp = 1;
static bool carRotEnable = false;
static bool running = true;
static std::bitset<1024> keystates;
static bool enablePerfStats = false;

static std::chrono::steady_clock::time_point lastTP;
static ScrollingBuffer lossBuf;
static ScrollingBuffer processedRayBuf;

#include "main_tools.h"

void setRenderMode() {
	std::string modeStr;
	switch (nrcRenderMode) {
		case ORIGINAL: modeStr = "ORIGINAL"; break;
		case REFERENCE: modeStr = "REFERENCE"; break;
		case NRC_PRIMARY: modeStr = "NRC_PRIMARY"; break;
		case NRC_FULL: modeStr = "NRC_FULL"; break;
	}
	renderer->SettingStringExt("nrcRenderMode", modeStr.c_str());
}

//  +-----------------------------------------------------------------------------+
//  |  PrepareScene                                                               |
//  |  Initialize a scene.                                                  LH2'21|
//  +-----------------------------------------------------------------------------+
void PrepareScene()
{
	// initialize scene
	renderer->AddScene( "scene.gltf", "../_shareddata/pica/" );
	renderer->SetNodeTransform( renderer->FindNode( "RootNode (gltf orientation matrix)" ), mat4::RotateX( -PI / 2 ) );
	int lightMat = renderer->AddMaterial( make_float3( 100, 100, 80 ) );
	int lightQuad = renderer->AddQuad( make_float3( 0, -1, 0 ), make_float3( 0, 26.0f, 0 ), 6.9f, 6.9f, lightMat );
	renderer->AddInstance( lightQuad );
	car = renderer->AddInstance( renderer->AddMesh( "legocar.obj", "../_shareddata/", 10.0f ) );
	carRotEnable = true;
}

// void PrepareSceneZeroDay() {
// 	renderer->AddScene("Measure_seven.glb", "../_shareddata/");
// }

void PrepareSceneDTS() {
	renderer->AddScene("DTS_nrc.glb", "../_shareddata/");
	//renderer->AddPointLight(make_float3(5.62, 1.22, -0.47), make_float3(100));
	//renderer->AddPointLight(make_float3(5.9098, -5.4423, 1.5712), make_float3(1));
	// renderer->AddPointLight(make_float3(5.92, 1.73, 6.09), make_float3(100));
}

//  +-----------------------------------------------------------------------------+
//  |  HandleInput                                                                |
//  |  Process user input.                                                  LH2'21|
//  +-----------------------------------------------------------------------------+
void HandleInput( float frameTime )
{
	// handle keyboard input
	float spd = (keystates[GLFW_KEY_LEFT_SHIFT] ? 15.0f : 5.0f) * frameTime, rot = 2.5f * frameTime;
	Camera* camera = renderer->GetCamera();
	if (keystates[GLFW_KEY_A]) camera->TranslateRelative( make_float3( -spd, 0, 0 ) );
	if (keystates[GLFW_KEY_D]) camera->TranslateRelative( make_float3( spd, 0, 0 ) );
	if (keystates[GLFW_KEY_W]) camera->TranslateRelative( make_float3( 0, 0, spd ) );
	if (keystates[GLFW_KEY_S]) camera->TranslateRelative( make_float3( 0, 0, -spd ) );
	if (keystates[GLFW_KEY_R]) camera->TranslateRelative( make_float3( 0, spd, 0 ) );
	if (keystates[GLFW_KEY_F]) camera->TranslateRelative( make_float3( 0, -spd, 0 ) );
	if (keystates[GLFW_KEY_UP]) camera->TranslateTarget( make_float3( 0, -rot, 0 ) );
	if (keystates[GLFW_KEY_DOWN]) camera->TranslateTarget( make_float3( 0, rot, 0 ) );
	if (keystates[GLFW_KEY_LEFT]) camera->TranslateTarget( make_float3( -rot, 0, 0 ) );
	if (keystates[GLFW_KEY_RIGHT]) camera->TranslateTarget( make_float3( rot, 0, 0 ) );
}

void DrawUI() {
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

	ImGui::Begin("NRC ExtSetting", 0);

	std::string fRendered = "FrameRendered: " + std::to_string(frameRendered);
	ImGui::Text(fRendered.c_str());

	auto current = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed_seconds = std::chrono::duration<double>(current - lastTP);
	double fps = 1.0 / elapsed_seconds.count();
	lastTP = current;
	ImGui::Text("FPS: %lf", fps);

	// RTSwitch
	{
		std::string hint = "CurrentRT: " + currentRT;
		ImGui::Text(hint.c_str());
		if (ImGui::Button("result")) {
			currentRT = "result";
		} else {
			for (size_t i = 0; i < auxRenderTargets.size(); i++) {
				auto &rt = auxRenderTargets[i];
				ImGui::PushID(i);
				if (ImGui::Button(rt.rtName.c_str())) {
					if (currentRT != "result") {
						renderer->SettingStringExt("clearAuxTargetInterest", currentRT.c_str());
					}
					currentRT = rt.rtName;
					renderer->SettingStringExt("setAuxTargetInterest", currentRT.c_str());
				}
				ImGui::SameLine();
				if (ImGui::Button(rt.accumulative ? "UnAccu" : "Accu")) {
					if (rt.accumulative)
						renderer->SettingStringExt("clearAuxTargetAccumulative", rt.rtName.c_str());
					else
						renderer->SettingStringExt("setAuxTargetAccumulative", rt.rtName.c_str());
					
					rt.accumulative = !rt.accumulative;
				}
				ImGui::PopID();
			}
		}
	}
	

	ImGui::Separator();

	// RenderMode switch
	{
		if (ImGui::Combo("render mode", (int*)&nrcRenderMode, "Original\0Reference\0NRCPrimary\0NRCFull\0")) {
			setRenderMode();
			frameRendered = 0;
			lossBuf.Erase();
			processedRayBuf.Erase();
		}
	}


	ImGui::Checkbox("Converge", &renderConverge);

	std::string samplesTaken = renderer->GetSettingStringExt("samplesTaken");
	ImGui::Text("samplesTaken: %s", samplesTaken.c_str());

	ImGui::Separator();

	if (nrcRenderMode != ORIGINAL && nrcRenderMode != REFERENCE && ImGui::Button("Reset network")) {
		renderer->SettingStringExt("nrcResetNet", "uniform");
	}

	if (nrcRenderMode != ORIGINAL && nrcRenderMode != REFERENCE && ImGui::Checkbox("Enable training", &nrcTrainingEnable)) {
		renderer->SettingStringExt(
			"nrcTrainingEnable",
			nrcTrainingEnable ? "true" : "false"
		);
	}

	// numInitialTrainingRays
	if (ImGui::DragInt("numInitialRays", &nrcNumInitialTrainingRays, 10.0f, 1, scrwidth * scrheight)) {
		// value changed, notify
		if (nrcNumInitialTrainingRays < 1) {
			nrcNumInitialTrainingRays = 1;
		}

		if (nrcNumInitialTrainingRays > scrwidth * scrheight) {
			nrcNumInitialTrainingRays = scrwidth * scrheight;
		}
		std::string rayStr = std::to_string(nrcNumInitialTrainingRays);
		renderer->SettingStringExt("nrcNumInitialTrainingRays", rayStr.c_str());
	}
	
	// TrainVisLayer
	if (ImGui::SliderInt("TrainVisLayer", &trainVisLayer, 0, nrcMaxTrainPathLength - 1)) {
		std::string tvLayer = std::to_string(trainVisLayer);
		bool success = renderer->SettingStringExt("trainVisLayer", tvLayer.c_str());
		if (!success) {
			trainVisLayer = 0;
		}
	}

	ImGui::Separator();

	// History buffer
	if (nrcRenderMode != ORIGINAL && nrcRenderMode != REFERENCE) {
		static int history = 10.0f;
		ImGui::SliderInt("Hist Disp Size", &history, 1, ScrollingBufferMaxSize - 1);

		if (lossBuf.Data.size() > 0 && ImPlot::BeginPlot("##Loss", ImVec2(-1,150))) {
			ImPlot::SetupAxes(
				NULL, NULL,
				ImPlotAxisFlags_NoTickLabels, ImPlotAxisFlags_AutoFit
			);
			ImPlot::SetupAxisLimits(ImAxis_X1, frameRendered - history, frameRendered, ImGuiCond_Always);
			ImPlot::SetupAxisLimits(ImAxis_Y1, 0, 1);
			ImPlot::SetNextFillStyle(IMPLOT_AUTO_COL, 0.5f);
			ImPlot::PlotLine("Loss", &lossBuf.Data[0].x, &lossBuf.Data[0].y, lossBuf.Data.size(), lossBuf.Offset, 2 * sizeof(float));
			ImPlot::EndPlot();
		}

		uint lastData = lossBuf.Data.size() < ScrollingBufferMaxSize ? 
			lossBuf.Data.size() - 1 : (lossBuf.Offset == 0 ? lossBuf.Data.size() - 1 : lossBuf.Offset - 1);

		if (lossBuf.Data.size() > 0) {
			ImGui::Text("Loss: %.2f", lossBuf.Data[lastData].y);
		}

		if (processedRayBuf.Data.size() > 0 && ImPlot::BeginPlot("##Rayprocessed", ImVec2(-1,150))) {
			ImPlot::SetupAxes(
				NULL, NULL,
				ImPlotAxisFlags_NoTickLabels, ImPlotAxisFlags_AutoFit
			);
			ImPlot::SetupAxisLimits(ImAxis_X1, frameRendered - history, frameRendered, ImGuiCond_Always);
			ImPlot::SetupAxisLimits(ImAxis_Y1, 0, 1);
			ImPlot::SetNextFillStyle(IMPLOT_AUTO_COL, 0.5f);
			ImPlot::PlotLine("Ray processed", &processedRayBuf.Data[0].x, &processedRayBuf.Data[0].y, processedRayBuf.Data.size(), processedRayBuf.Offset, 2 * sizeof(float));
			ImPlot::EndPlot();
		}

		if (processedRayBuf.Data.size() > 0) {
			ImGui::Text("Ray processed: %.0f", processedRayBuf.Data[lastData].y);
		}
	}

	ImGui::End();

	// Cam parameters
	{
		ImGui::Begin( "Camera parameters", 0 );
		float3 camPos = renderer->GetCamera()->transform.GetTranslation();
		float3 camDir = renderer->GetCamera()->transform.GetForward();
		ImGui::Text( "position: %5.2f, %5.2f, %5.2f", camPos.x, camPos.y, camPos.z );
		ImGui::Text( "viewdir:  %5.2f, %5.2f, %5.2f", camDir.x, camDir.y, camDir.z );
		ImGui::SliderFloat( "FOV", &renderer->GetCamera()->FOV, 10, 90 );
		ImGui::SliderFloat( "aperture", &renderer->GetCamera()->aperture, 0, 0.025f );
		ImGui::SliderFloat( "distortion", &renderer->GetCamera()->distortion, 0, 0.5f );
		ImGui::SliderFloat( "focalDistance", &renderer->GetCamera()->focalDistance, 0, 100.0f );
		ImGui::SliderFloat( "aspectRatio", &renderer->GetCamera()->aspectRatio, 0.1f, 10.0f );
		ImGui::Combo( "tonemap", &renderer->GetCamera()->tonemapper, "clamp\0reinhard\0reinhard ext\0reinhard lum\0reinhard jodie\0uncharted2\0\0" );
		ImGui::SliderFloat( "brightness", &renderer->GetCamera()->brightness, 0, 0.5f );
		ImGui::SliderFloat( "contrast", &renderer->GetCamera()->contrast, 0, 0.5f );
		ImGui::SliderFloat( "gamma", &renderer->GetCamera()->gamma, 1, 2.5f );
		ImGui::End();
	}

	// Query perf-stat string
	{
		ImGui::Begin("Performance Statistics", 0);

		// NOTE: may generate lots of cuda error invalid resource handle
		// while calling to determine elapsed time
		// These type of errors are generally non-sticky, but may
		// break GEMM codes since they may fail to check & a problem
		// for compute-santizer based debugging
		// So, open only if users requested
		ImGui::Checkbox("Perf Stats", &enablePerfStats);

		if (enablePerfStats) {
			std::string perfData = renderer->GetSettingStringExt("perfStats");
			ImGui::Text("%s", perfData.c_str());
		}
		ImGui::End();
	}
	

	//ImPlot::ShowDemoWindow();

	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData( ImGui::GetDrawData() );
}

void InitImGui()
{
	// Setup Dear ImGui context
	IMGUI_CHECKVERSION();
	if (!ImGui::CreateContext()) {
		printf("ImGui::CreateContext failed.\n");
		exit(EXIT_FAILURE);
	}

	if (!ImPlot::CreateContext()) {
		printf("ImPlot::CreateContext failed.\n");
		exit(EXIT_FAILURE);
	}

	ImGuiIO& io = ImGui::GetIO(); (void)io;
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
	ImGui::StyleColorsDark(); // or ImGui::StyleColorsClassic();
	ImGui_ImplGlfw_InitForOpenGL( window, true );
	ImGui_ImplOpenGL3_Init( "#version 130" );
}

void InitAuxRT() {
	currentRT = "result";

	string numRays = renderer->GetSettingStringExt("nrcNumInitialTrainingRays");
	nrcNumInitialTrainingRays = std::atoi(numRays.c_str());

	auxRTEnabled = renderer->EnableFeatureExt("auxiliaryRenderTargets");
	if (!auxRTEnabled) return;

	string maxTrainPathLen = renderer->GetSettingStringExt("nrcMaxTrainPathLength");
	nrcMaxTrainPathLength = std::atoi(maxTrainPathLen.c_str());
}

//  +-----------------------------------------------------------------------------+
//  |  main                                                                       |
//  |  Application entry point.                                             LH2'21|
//  +-----------------------------------------------------------------------------+
int main()
{
	// Initialize staticically linked FreeImage
	FreeImage_Initialise(FALSE);

	// initialize OpenGL
	InitGLFW();
	InitImGui();
	// initialize renderer
	renderer = RenderAPI::CreateRenderAPI( "RenderCore_Optix7NRC" );
	// renderer = RenderAPI::CreateRenderAPI( "RenderCore_Optix7" );
	renderer->DeserializeCamera( "camera.xml" );
	// initialize auxiliary rendertargets
	InitAuxRT();
	// initialize nrc render mode - todo: duplicate apply
	// nrcRenderMode = nrcRenderModeSet::ORIGINAL;
	// nrcRenderMode = nrcRenderModeSet::NRC_PRIMARY;
	nrcRenderMode = nrcRenderModeSet::NRC_FULL;
	setRenderMode();
	// initialize scene
	// PrepareScene();
	// PrepareSceneZeroDay();
	PrepareSceneDTS();
	// set initial window size
	ReshapeWindowCallback( 0, SCRWIDTH, SCRHEIGHT );
	// enter main loop
	while (!glfwWindowShouldClose( window ))
	{
		// update scene
		renderer->SynchronizeSceneData();
		// render
		renderer->Render( renderConverge ? Converge : Restart );
		frameRendered++;

		// Scrolling buffer maintenance
		if (nrcRenderMode != ORIGINAL && nrcRenderMode != REFERENCE) {
			string lastLossStr = renderer->GetSettingStringExt("lastLoss");
			float lastLoss = std::atof(lastLossStr.c_str());

			string lastProcessedRaysStr = renderer->GetSettingStringExt("lastProcessedRays");
			int lastProcessedRays = std::atoi(lastProcessedRaysStr.c_str());

			lossBuf.AddPoint(frameRendered, lastLoss);
			processedRayBuf.AddPoint(frameRendered, lastProcessedRays);
		}

		// handle user input
		HandleInput( 0.025f );

		if (carRotEnable) {
			// minimal rigid animation example
			static float r = 0;
			mat4 M = mat4::RotateY( r * 2.0f ) * mat4::RotateZ( 0.2f * sinf( r * 8.0f ) ) * mat4::Translate( make_float3( 0, 5, 0 ) );
			renderer->SetNodeTransform( car, M );
			if ((r += 0.025f * 0.3f) > 2 * PI) r -= 2 * PI;
		}
		
		// finalize and present
		shader->Bind();
		
		if (currentRT == "result") {
			shader->SetInputTexture( 0, "color", renderTarget );
		} else {
			bool rtFound = false;
			for (auto &auxRT: auxRenderTargets) {
				if (auxRT.rtName == currentRT) {
					shader->SetInputTexture(0, "color", auxRT.texture.get());
					rtFound = true;
					break;
				}
			}
			assert(rtFound);
		}
		shader->SetInputMatrix( "view", mat4::Identity() );
		DrawQuad();
		shader->Unbind();

		// shader
		DrawUI();

		// finalize
		glfwSwapBuffers( window );
		glfwPollEvents();
		if (!running) break; // esc was pressed
	}
	// clean up
	disableAllAuxRT();
	renderer->SerializeCamera( "camera.xml" );
	renderer->Shutdown();
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImPlot::DestroyContext();
	ImGui::DestroyContext();
	glfwDestroyWindow( window );
	glfwTerminate();

	// Deinitialize staticically linked FreeImage
	FreeImage_DeInitialise();
	return 0;
}

// EOF