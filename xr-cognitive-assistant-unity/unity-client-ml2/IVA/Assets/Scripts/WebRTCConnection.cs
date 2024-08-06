using UnityEngine.InputSystem;

using MagicLeap.Core;
using SimpleJson;
using System;
using System.Linq;
using System.Collections;
using System.Collections.Generic;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.SceneManagement;
using UnityEngine.UI;
using UnityEngine.XR.MagicLeap;


/// <summary>
/// This class handles video recording and loading based on controller
/// input.
/// </summary>
public class WebRTCConnection : MonoBehaviour
{

    class AcceptAnyCertificate : CertificateHandler {
        protected override bool ValidateCertificate(byte[] certificateData) => true;
    }

    [SerializeField, Tooltip("Desired width for the camera capture")]
    private int captureWidth = 1280;

    [SerializeField, Tooltip("Desired height for the camera capture")]
    private int captureHeight = 720;
    private readonly string LLM_OUTPUT_DC_LABEL = "llm_output_dc";

    [SerializeField]
    private string serverAddress = "";
    [SerializeField]
    private Text resultText;

    [SerializeField]
    private MeshingSubsystemComponent _meshingSubsystemComponent = null;

    public MLWebRTC.MediaStream.Track.AudioType audioType = MLWebRTC.MediaStream.Track.AudioType.Microphone;
    public MLWebRTCLocalAppDefinedAudioSourceBehavior localAppDefinedAudioSourceBehavior;
    public MLWebRTCAudioSinkBehavior remoteAudioSinkBehavior;
    private MLWebRTC.PeerConnection connection = null;
    private MLWebRTC.DataChannel dataChannel = null;
    private bool dataChannelOpened = false;
    private MLWebRTC.MediaStream localMediaStream = null;
    private MLWebRTC.MediaStream remoteMediaStream = null;
    private DefinedAudioSourceExample localDefinedAudioSource;
    private MLCamera.ConnectFlag selectedFlag = MLCamera.ConnectFlag.CamOnly;
    public GameObject cubePrefab;
    private string current_coords = null;
    private bool sent = false;
    private int frameCount = 0;

    private List<GameObject> cubes = new List<GameObject>();

    private bool isCameraConnected;
    private MLCamera.StreamCapability selectedCapability;

    private MagicLeapInputs mlInputs;
    private MagicLeapInputs.ControllerActions controllerActions;

    private MLCamera colorCamera;
    private bool cameraDeviceAvailable = false;
    private bool isCapturing;
    private string poseText;

    private readonly MLPermissions.Callbacks permissionCallbacks = new MLPermissions.Callbacks();
    private static readonly string[] requiredPermissions = new string[] { MLPermission.Camera, MLPermission.RecordAudio };
    private readonly HashSet<string> grantedPermissions = new HashSet<string>();

    private IDictionary<int, MLCamera.IntrinsicCalibrationParameters> instrinsics = new Dictionary<int, MLCamera.IntrinsicCalibrationParameters>();
    private IDictionary<int, Matrix4x4> cameraPose = new Dictionary<int, Matrix4x4>();
    
    private int frameID = 0;


    private Coroutine enableCameraCoroutine;
    
    /// <summary>
    /// Using Awake so that Permissions is set before PermissionRequester Start.
    /// </summary>
    void Awake()
    {

        isCapturing = false;

        permissionCallbacks.OnPermissionGranted += OnPermissionGranted;
        permissionCallbacks.OnPermissionDenied += OnPermissionDenied;
        permissionCallbacks.OnPermissionDeniedAndDontAskAgain += OnPermissionDenied;

        foreach (string permission in requiredPermissions)
        {
            MLPermissions.RequestPermission(permission, permissionCallbacks);
        }
    }

    /// <summary>
    /// Stop the camera, unregister callbacks, and stop input and permissions APIs.
    /// </summary>
    void OnDisable()
    {
        permissionCallbacks.OnPermissionGranted -= OnPermissionGranted;
        permissionCallbacks.OnPermissionDenied -= OnPermissionDenied;
        permissionCallbacks.OnPermissionDeniedAndDontAskAgain -= OnPermissionDenied;
        if (colorCamera != null && isCameraConnected)
        {
            DisableMLCamera();
        }
        // controllerActions.Bumper.performed -= OnButtonDown;
        mlInputs.Dispose();

    }

    /// <summary>
    /// Display permission error if necessary or update status text.
    /// </summary>
    private void Update()
    {
        //UpdateStatusText();
    }

    

    private void CheckAndStopPreviousCoroutine()
    {
        if (enableCameraCoroutine == null) return;
        StopCoroutine(enableCameraCoroutine);
        enableCameraCoroutine = null;
    }
    
    private void OnApplicationPause(bool pauseStatus)
    {
        if (pauseStatus)
        {
            colorCamera.OnRawVideoFrameAvailable -= OnCaptureRawVideoFrameAvailable;
        }
        else
        {
            colorCamera.OnRawVideoFrameAvailable += OnCaptureRawVideoFrameAvailable;
        }
    }

    /// <summary>
    /// Captures a still image using the device's camera and returns
    /// the data path where it is saved.
    /// </summary>
    /// <param name="fileName">The name of the file to be saved to.</param>
    private void StartVideoCapture()
    {
        MLCamera.OutputFormat outputFormat = MLCamera.OutputFormat.RGBA_8888;
        MLCamera.CaptureConfig captureConfig = new MLCamera.CaptureConfig();
        captureConfig.CaptureFrameRate = MLCamera.CaptureFrameRate._15FPS;
        captureConfig.StreamConfigs = new MLCamera.CaptureStreamConfig[1];
        captureConfig.StreamConfigs[0] = MLCamera.CaptureStreamConfig.Create(selectedCapability, outputFormat);
        MLResult result = colorCamera.PrepareCapture(captureConfig, out MLCamera.Metadata _);
        if (result.IsOk)
        {
            result = colorCamera.PreCaptureAEAWB();
            result = colorCamera.CaptureVideoStart();
            if (!result.IsOk)
            {
                Debug.LogError("Failed to start video capture!");
            }
            
        }

        isCapturing = result.IsOk;
    }

    private void StopVideoCapture()
    {
        if (isCapturing)
        {
            colorCamera.CaptureVideoStop();


        }
        
        isCapturing = false;
    }

    /// <summary>
    /// Connects the MLCamera component and instantiates a new instance
    /// if it was never created.
    /// </summary>
    private IEnumerator EnableMLCamera()
    {
        
        if (colorCamera != null)
        {
            yield return null;
        }

        while (!cameraDeviceAvailable)
        {
            MLResult result = MLCamera.GetDeviceAvailabilityStatus(MLCamera.Identifier.CV, out cameraDeviceAvailable);
            if (!(result.IsOk && cameraDeviceAvailable))
            {
                // Wait until camera device is available
                yield return new WaitForSeconds(1.0f);
            }
        }

        Debug.Log("Camera device available");
        yield return new WaitForSeconds(1.0f);

        MLCamera.ConnectContext context = MLCamera.ConnectContext.Create();
        context.EnableVideoStabilization = true;
        context.CamId = MLCamera.Identifier.CV;

        colorCamera = MLCamera.CreateAndConnect(context);
        if (colorCamera != null)
        {
            Debug.Log("Camera device connected");
            isCameraConnected = true;
            MLCamera.StreamCapability[] streamCapabilities = MLCamera.GetImageStreamCapabilitiesForCamera(colorCamera, MLCamera.CaptureType.Video);
            if (streamCapabilities == null || streamCapabilities.Length <= 0)
            {
                Debug.LogError("Camera device unable to received stream caps.");
                yield break;
            }

            if (!MLCamera.TryGetBestFitStreamCapabilityFromCollection(streamCapabilities, captureWidth, captureHeight,
                MLCamera.CaptureType.Video, out selectedCapability))
            {
                Debug.LogError("Camera device unable to fit stream caps to chosen options.");
                yield break;
            }

            Debug.Log("Camera device received stream caps");
            colorCamera.OnRawVideoFrameAvailable += OnCaptureRawVideoFrameAvailable;
            StartVideoCapture();
            string id = $"local{selectedFlag}";
            localMediaStream = MLWebRTC.MediaStream.CreateWithBuiltInTracks(id, MLWebRTC.MediaStream.Track.VideoType.None, MLWebRTC.MediaStream.Track.AudioType.Microphone, "", id);
            startConnection();
            InitTracks();
            CreateDataChannel();
            CreateOffer();
        }
    }

    private void InitTracks()
    {
        if (localAppDefinedAudioSourceBehavior != null)
        {
            if (audioType == MLWebRTC.MediaStream.Track.AudioType.Defined)
            {
                localAppDefinedAudioSourceBehavior.gameObject.SetActive(true);
                localAppDefinedAudioSourceBehavior.Init(localMediaStream.AudioTracks[0] as DefinedAudioSourceExample);
            }
        }
        if(!connection.ContainsTrack(localMediaStream.ActiveAudioTrack))
            connection.AddLocalTrack(localMediaStream.ActiveAudioTrack);

    }

    private void startConnection(){
        connection = MLWebRTC.PeerConnection.CreateRemote(CreateIceServers(), out MLResult result);
        if (!result.IsOk)
        {
            Debug.LogFormat("MLWebRTCExample.Login failed to create a connection. Reason: {0}.", MLResult.CodeToString(result.Result));
            return;
        }

        connection.OnError += OnConnectionError;
        connection.OnConnected += OnConnectionConnected;
        connection.OnDisconnected += OnConnectionDisconnected;
        connection.OnTrackAddedMultipleStreams += OnConnectionTrackAdded;
        connection.OnTrackRemovedMultipleStreams += OnConnectionTrackRemoved;
        connection.OnDataChannelReceived += OnConnectionDataChannelReceived;
        connection.OnLocalOfferCreated += OnConnectionLocalOfferCreated;
        connection.OnLocalAnswerCreated += OnConnectionLocalAnswerCreated;
    
    }
        private void OnConnectionError(MLWebRTC.PeerConnection connection, string errorMessage)
    {
        Debug.Log(errorMessage);
    }

    private void OnConnectionConnected(MLWebRTC.PeerConnection connection)
    {
        Debug.Log("Connected!");
    }

    private void OnConnectionDisconnected(MLWebRTC.PeerConnection connection)
    {
        Debug.Log("Disconnected!");
    }

    private void OnConnectionTrackAdded(List<MLWebRTC.MediaStream> mediaStreams, MLWebRTC.MediaStream.Track addedTrack)
    {
        // Debug.Log($"Adding {addedTrack.TrackType} track.");
        if (remoteMediaStream == null)
        {
            remoteMediaStream = mediaStreams[0];
        }

        switch (addedTrack.TrackType)
        {
            // if the incoming track is audio, set the audio sink to this track.
            case MLWebRTC.MediaStream.Track.Type.Audio:
                remoteAudioSinkBehavior.AudioSink.SetStream(remoteMediaStream);
                remoteAudioSinkBehavior.gameObject.SetActive(true);
                remoteAudioSinkBehavior.AudioSink.SetCacheSize(200);
                break;

        }
    }
    private void OnConnectionTrackRemoved(List<MLWebRTC.MediaStream> mediaStream, MLWebRTC.MediaStream.Track removedTrack)
    {
        Debug.Log($"Removed {removedTrack.TrackType} track.");

        switch (removedTrack.TrackType)
        {
            case MLWebRTC.MediaStream.Track.Type.Audio:
                remoteAudioSinkBehavior.AudioSink.SetStream(null);
                remoteAudioSinkBehavior.gameObject.SetActive(false);
                break;

        }
    }

    private void OnConnectionDataChannelReceived(MLWebRTC.PeerConnection connection, MLWebRTC.DataChannel dataChannel)
    {
        Debug.Log("Data channel received!");
    }

    public MLWebRTC.IceServer[] CreateIceServers()
    {
        string stunServerUri = "stun:stun.l.google.com:19302";

        MLWebRTC.IceServer[] iceServers = new MLWebRTC.IceServer[1];

        iceServers[0] = MLWebRTC.IceServer.Create(stunServerUri);

        return iceServers;
    }

    public static string FormatSdpOffer(string offer, string sdp)
    {
        JsonObject jsonObj = new JsonObject();
        jsonObj["sdp"] = sdp;
        jsonObj["type"] = offer;
        return jsonObj.ToString();
    }

    private bool ParseAnswer(string data, out string sdp)
    {
        bool result = false;
        sdp = "";

        if (data == "{}" || data == string.Empty)
        {
            return result;
        }

        SimpleJson.SimpleJson.TryDeserializeObject(data, out object obj);
        if (obj == null)
        {
            return false;
        }

        JsonObject jsonObj = (JsonObject)obj;
        if (jsonObj.ContainsKey("sdp") && jsonObj.ContainsKey("type"))
        {
            sdp = (string)jsonObj["sdp"];
            result = true;
        }

        return result;
    }

    private void OnConnectionLocalOfferCreated(MLWebRTC.PeerConnection connection, string sendSdp)
    {
        if (serverAddress == "")
        {
            Debug.LogError("Server address is empty!");
            return;
        }
        Debug.Log($"Sending offer to {serverAddress}...");
        StartCoroutine(SendOffer(sendSdp));
    }

    private void SubscribeToDataChannel(MLWebRTC.DataChannel dataChannel)
    {
        dataChannel.OnClosed += OnDataChannelClosed;
        dataChannel.OnOpened += OnDataChannelOpened;
        dataChannel.OnMessageText += OnDataChannelTextMessage;
        dataChannel.OnMessageBinary += OnDataChannelBinaryMessage;
    }

    private void UnsubscribeFromDataChannel(MLWebRTC.DataChannel dataChannel)
    {
        dataChannel.OnClosed -= OnDataChannelClosed;
        dataChannel.OnOpened -= OnDataChannelOpened;
        dataChannel.OnMessageText -= OnDataChannelTextMessage;
        dataChannel.OnMessageBinary -= OnDataChannelBinaryMessage;
    }

    private void OnDataChannelOpened(MLWebRTC.DataChannel dataChannel)
    {
        Debug.Log("Data channel opened!");
        dataChannelOpened = true;
    }

    private void OnDataChannelClosed(MLWebRTC.DataChannel dataChannel)
    {
        Debug.Log("Data channel closed!");
        UnsubscribeFromDataChannel(dataChannel);
    }

    private void OnDataChannelTextMessage(MLWebRTC.DataChannel dataChannel, string message)
    {
        Debug.Log(message);
        Debug.Log(message.Contains("///"));
        if (message.Contains("///")){
            string[] idMessage = message.Split("///");
            int currFrame = Int32.Parse(idMessage[0]);
            MLCamera.IntrinsicCalibrationParameters currInstrinsics = instrinsics[currFrame];
            Matrix4x4 currCameraPose = cameraPose[currFrame];
            Debug.Log(currFrame);

            string[] coords = idMessage[1].Split("coords=");
            try
                {
                    foreach(GameObject cube in cubes) {
                        Destroy(cube);
                    }
                    current_coords = coords[1];

                    // Parse the JSON string
                    object parsedObject = SimpleJson.SimpleJson.DeserializeObject(current_coords);

                    // Convert the parsed object to a list of integers
                    
                    if (parsedObject is List<object> outerList)
                    {
                        
                        foreach (object innerObj in outerList)
                        {
                            List<int> integers = new List<int>();
                            if (innerObj is List<object> innerList)
                            {
                                foreach (object numObj in innerList)
                                {
                                    if (numObj is long num)
                                    {
                                        integers.Add((int)num);
                                    }
                                }
                                GameObject cube = Instantiate(cubePrefab);
                                cubes.Add(cube);
                                Debug.Log("Created cube \n");
                                Vector2 pointin2D = new Vector2(integers[0], integers[1]);
                                Vector3 pointIn3D = CameraUtilities.CastRayFromScreenToWorldPoint(currInstrinsics, currCameraPose, pointin2D);
                                pointIn3D = new Vector3(pointIn3D[0], (float) (pointIn3D[1] - 0.4), pointIn3D[2]); 
                                cube.transform.position = pointIn3D; //center of object - y value
                            }
                        }
                    // Output the list of integers
                        // Debug.Log("List of integers:");
                        // foreach (int num in integers)
                        // {
                        //     Debug.Log(num);
                        // }
                        
                    }
    
                }
                catch (Exception e)
                {
                    current_coords = null;
                }
            Debug.Log(current_coords);
        }
        else{
            resultText.text = "=== Message Received From Server ===\n" + message;
        }
        
    }


    private void OnDataChannelBinaryMessage(MLWebRTC.DataChannel dataChannel, byte[] message)
    {

    }

    private IEnumerator SendOffer(string sendSdp)
    {
        var cert = new AcceptAnyCertificate();
        UnityWebRequest webRequest = UnityWebRequest.Post($"{serverAddress}/offer", FormatSdpOffer("offer", sendSdp), "application/json");
        webRequest.certificateHandler = cert;
        using (webRequest)
        {
            yield return webRequest.SendWebRequest();

            if (webRequest.result != UnityWebRequest.Result.Success)
            {
                Debug.Log(webRequest.error);
            }
            else
            {
                string response = webRequest.downloadHandler.text;
                if (ParseAnswer(response, out string remoteAnswer))
                {
                    Debug.Log("Got answer!");
                    connection.SetRemoteAnswer(remoteAnswer);
                    // connection.SetRemoteOffer(remoteAnswer);
                }
            }
        }
    }

    private void OnConnectionLocalAnswerCreated(MLWebRTC.PeerConnection connection, string sendAnswer)
    {
        // Debug.Log("Sending answer to an offer...");
        // UnityWebRequest webRequest = UnityWebRequest.Post($"{serverAddress}/answer", FormatSdpOffer("answer", sendAnswer), "application/json");
        // webRequest.SendWebRequest();
    }

    public void SendMessageOnDataChannel(string message)
    {
        if (!this.dataChannelOpened)
            return;

        if (string.IsNullOrEmpty(message))
            return;

        MLResult? result = this.dataChannel?.SendMessage(message);
        if (result.HasValue)
        {
            if (result.Value.IsOk)
            {
                Debug.Log("Sent: " + message);
            }
            else
            {
                Debug.LogError($"MLWebRTC.DataChannel.SendMessage() failed with error {result}");
            }
        }
    }

    /// <summary>
    /// Disconnects the MLCamera if it was ever created or connected.
    /// </summary>
    private void DisableMLCamera()
    {
        if (colorCamera != null)
        {
            colorCamera.OnRawVideoFrameAvailable -= OnCaptureRawVideoFrameAvailable;
            colorCamera.Disconnect();
            // Explicitly set to false here as the disconnect was attempted.
            isCameraConnected = false;
            colorCamera = null;
        }
    }
    

    /// <summary>
    /// Handles the event for button down.
    /// </summary>
    /// <param name="controllerId">The id of the controller.</param>
    /// <param name="button">The button that is being pressed.</param>
    private void OnButtonDown(InputAction.CallbackContext obj)
    {
        if (colorCamera is { IsPaused: true } || !isCameraConnected)
        {
            return;
        }
        
        if (!isCapturing)
        {
            StartVideoCapture();
        }
        else
        {
            StopVideoCapture();
        }
    }

    /// <summary>
    /// Handles the event of a new image getting captured.
    /// </summary>
    /// <param name="imageData">The raw data of the image.</param>
    private void OnCaptureRawVideoFrameAvailable(MLCamera.CameraOutput capturedFrame, MLCamera.ResultExtras resultExtras, MLCamera.Metadata metadataHandle)
    {
        OnCaptureDataReceived(resultExtras, capturedFrame);
    // if (MLCVCamera.GetFramePose(resultExtras.VCamTimestamp, out Matrix4x4 cameraTransform).IsOk)
    // {
    //     uint width = capturedFrame.Planes[0].Width;
    //     uint height = capturedFrame.Planes[0].Height;

    //     Vector2 topLeftPixel = new Vector2(0, 0);
    //     Vector2 topRightPixel = new Vector2(width, 0);
    //     Vector2 bottomLeftPixel = new Vector2(0, height);
    //     Vector2 bottomRightPixel = new Vector2(width, height);
    //     Vector2 centerPixel = new Vector2(width / 2f, height / 2f);

    //     Vector3 TopLeftObject = CameraUtilities.CastRayFromScreenToWorldPoint(resultExtras.Intrinsics.Value, cameraTransform,topLeftPixel);
    //     Vector3 TopRightObject = CameraUtilities.CastRayFromScreenToWorldPoint(resultExtras.Intrinsics.Value, cameraTransform, topRightPixel);
    //     Vector3 BottomLeftObject = CameraUtilities.CastRayFromScreenToWorldPoint(resultExtras.Intrinsics.Value, cameraTransform, bottomLeftPixel);
    //     Vector3 BottomRightObject = CameraUtilities.CastRayFromScreenToWorldPoint(resultExtras.Intrinsics.Value, cameraTransform, bottomRightPixel);
    //     Vector3 CenterObject = CameraUtilities.CastRayFromScreenToWorldPoint(resultExtras.Intrinsics.Value, cameraTransform, centerPixel);
    // }
    }

    private void OnPermissionDenied(string permission)
    {
        MLPluginLog.Error($"{permission} denied, example won't function.");
    }

    private void OnPermissionGranted(string permission)
    {
        grantedPermissions.Add(permission);
        if (grantedPermissions.Count == requiredPermissions.Length)
        {
            enableCameraCoroutine = StartCoroutine(EnableMLCamera());
        }
    }
    private void CreateDataChannel() {
        this.dataChannel = MLWebRTC.DataChannel.CreateLocal(connection, out MLResult result, LLM_OUTPUT_DC_LABEL);
        SubscribeToDataChannel(this.dataChannel);
    }
    private void CreateOffer()
    {
        connection.CreateOffer();
    }
    private void OnCaptureDataReceived(MLCamera.ResultExtras extras, MLCamera.CameraOutput frameData){
        if (dataChannelOpened){
            Debug.Log("in data received");
            if (frameData.Format == MLCamera.OutputFormat.RGBA_8888)
            {
                if (frameCount == 0){
                    if (MLCVCamera.GetFramePose(extras.VCamTimestamp, out Matrix4x4 cameraTransform).IsOk)
                    {
                        instrinsics.Add(frameID,extras.Intrinsics.Value);
                        cameraPose.Add(frameID, cameraTransform);
                        UploadRGBTexture(frameID, frameData.Planes[0]);
                        frameID++;
                    }
                }
                frameCount++;
                if (frameCount == 15){
                    frameCount = 0;
                }

            }
        }
    }

    private void UploadRGBTexture(int frameID, MLCamera.PlaneInfo imagePlane)
        {
            
            Debug.Log("in upload texture");
            byte[] bytesID =  BitConverter.GetBytes(frameID);
            int actualWidth = (int)(imagePlane.Width * imagePlane.PixelStride);
            
            
            var newTextureChannel = new byte[actualWidth * imagePlane.Height];
            // Debug.Log(imagePlane.Width);
            // Debug.Log(imagePlane.Height);
            // Debug.Log(imagePlane.PixelStride);
            for(int i = 0; i < imagePlane.Height; i++)
                {
                //Buffer.BlockCopy(imagePlane.Data, (int)0, newTextureChannel, (int)0, (int)imagePlane.Size);
                Buffer.BlockCopy(imagePlane.Data, (int)(i * imagePlane.Stride), newTextureChannel, i * actualWidth, actualWidth);
            }
            byte[] bytesToSend =  bytesID.Concat(newTextureChannel).ToArray();
            // System.Random rand = new System.Random();
            // byte[] randomIntegers = new byte[1228800];
            // for (int i = 0; i < randomIntegers.Length; ++i)
            // {
            //     randomIntegers[i] = (byte)rand.Next(0, 101);
            // }
        
            // MLResult? result = this.dataChannel?.SendMessage<byte>(newTextureChannel);
            // if (result.HasValue)
            // {
            //     if (result.Value.IsOk)
            //     {
            //         Debug.Log("Sent");
            //     }
            //     else
            //     {
            //         Debug.LogError($"MLWebRTC.DataChannel.SendMessage() failed with error {result}");
            //     }
            // }
            // var cert = new AcceptAnyCertificate();
            // UnityWebRequest webRequest = UnityWebRequest.Put($"{serverAddress}/image", newTextureChannel);
            // webRequest.certificateHandler = cert;
            // using (webRequest)
            // {
            //     yield return webRequest.SendWebRequest();

            //     if (webRequest.result != UnityWebRequest.Result.Success)
            //     {
            //         Debug.Log(webRequest.error);
            //     }
                
            // }
            StartCoroutine(Upload(bytesToSend));
            
        }

        IEnumerator Upload(byte[] b) {
        var cert = new AcceptAnyCertificate();
                UnityWebRequest webRequest = UnityWebRequest.Put($"{serverAddress}/image", b);
                webRequest.certificateHandler = cert;
                yield return webRequest.Send();
                Debug.Log("Sent");
        
                
        }
}

