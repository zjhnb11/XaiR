// %BANNER_BEGIN%
// ---------------------------------------------------------------------
// %COPYRIGHT_BEGIN%
// Copyright (c) (2019-2022) Magic Leap, Inc. All Rights Reserved.
// Use of this file is governed by the Software License Agreement, located here: https://www.magicleap.com/software-license-agreement-ml2
// Terms and conditions applicable to third-party materials accompanying this distribution may also be found in the top-level NOTICE file appearing herein.
// %COPYRIGHT_END%
// ---------------------------------------------------------------------
// %BANNER_END%

using System;
using UnityEngine;
using UnityEngine.XR.MagicLeap;
using SimpleJson;

namespace MagicLeap.Examples
{
    /// <summary>
    /// This class allows you to change meshing properties at runtime, including the rendering mode.
    /// Manages the MeshingSubsystemComponent behaviour and tracks the meshes.
    /// </summary>
    public class MeshingVisualizer : MonoBehaviour
    {
        public enum RenderMode
        {
            None,
            Wireframe,
            Colored,
            PointCloud,
            Occlusion
        }

        [SerializeField, Tooltip("The MeshingSubsystemComponent from which to get update on mesh types.")]
        private MeshingSubsystemComponent _meshingSubsystemComponent = null;

        [SerializeField, Tooltip("The material to apply for occlusion.")]
        private Material _occlusionMaterial = null;

        [SerializeField, Tooltip("The material to apply for wireframe rendering.")]
        private Material _wireframeMaterial = null;

        [SerializeField, Tooltip("The material to apply for colored rendering.")]
        private Material _coloredMaterial = null;

        [SerializeField, Tooltip("The material to apply for point cloud rendering.")]
        private Material _pointCloudMaterial = null;

        private Camera _camera = null;


        [SerializeField, Space, Tooltip("Render mode to render mesh data with.")]
        private MeshingVisualizer.RenderMode _renderMode = MeshingVisualizer.RenderMode.Wireframe;

        public RenderMode renderMode
        {
            get; private set;
        } = RenderMode.Wireframe;
        private readonly MLPermissions.Callbacks permissionCallbacks = new MLPermissions.Callbacks();
        public Transform raycastOrigin;


        /// <summary>
        /// Start listening for MeshingSubsystemComponent events.
        /// </summary>
        void Awake()
        {
            permissionCallbacks.OnPermissionGranted += OnPermissionGranted;
            permissionCallbacks.OnPermissionDenied += OnPermissionDenied;
            permissionCallbacks.OnPermissionDeniedAndDontAskAgain += OnPermissionDenied;
            _camera = Camera.main;
            
            // Validate all required game objects.
            if (_meshingSubsystemComponent == null)
            {
                Debug.LogError("Error: MeshingVisualizer._meshingSubsystemComponent is not set, disabling script!");
                enabled = false;
                return;
            }
            if (_occlusionMaterial == null)
            {
                Debug.LogError("Error: MeshingVisualizer._occlusionMaterial is not set, disabling script!");
                enabled = false;
                return;
            }
            if (_wireframeMaterial == null)
            {
                Debug.LogError("Error: MeshingVisualizer._wireframeMaterial is not set, disabling script!");
                enabled = false;
                return;
            }
            if (_coloredMaterial == null)
            {
                Debug.LogError("Error: MeshingVisualizer._coloredMaterial is not set, disabling script!");
                enabled = false;
                return;
            }
            if (_pointCloudMaterial == null)
            {
                Debug.LogError("Error: MeshingVisualizer._pointCloudMaterial is not set, disabling script!");
                enabled = false;
                return;
            }
        }

        /// <summary>
        /// Register for new and updated fragments.
        /// </summary>
        void Start()
        {
            Debug.Log("Started visualizer.");
            MLPermissions.RequestPermission(MLPermission.SpatialMapping, permissionCallbacks);
            _meshingSubsystemComponent.meshAdded += HandleOnMeshReady;
            _meshingSubsystemComponent.meshUpdated += HandleOnMeshReady;

            _meshingSubsystemComponent.gameObject.transform.position = _camera.gameObject.transform.position;
        }

        /// <summary>
        /// Unregister callbacks.
        /// </summary>
        void OnDestroy()
        {
            permissionCallbacks.OnPermissionGranted -= OnPermissionGranted;
            permissionCallbacks.OnPermissionDenied -= OnPermissionDenied;
            permissionCallbacks.OnPermissionDeniedAndDontAskAgain -= OnPermissionDenied;

            _meshingSubsystemComponent.meshAdded -= HandleOnMeshReady;
            _meshingSubsystemComponent.meshUpdated -= HandleOnMeshReady;
        }


        private void OnPermissionGranted(string permission)
        {
            _meshingSubsystemComponent.enabled = true;
            Debug.Log("Permission Granted.");
        }

        private void OnPermissionDenied(string permission)
        {
            Debug.LogError($"Failed to create Meshing Subsystem due to missing or denied {MLPermission.SpatialMapping} permission. Please add to manifest. Disabling script.");
            enabled = false;
            _meshingSubsystemComponent.enabled = false;
        }

        /// <summary>
        /// Set the render material on the meshes.
        /// </summary>
        /// <param name="mode">The render mode that should be used on the material.</param>
        public void SetRenderers(RenderMode mode)
        {
            if (renderMode != mode)
            {
                // Set the render mode.
                renderMode = mode;

                _meshingSubsystemComponent.requestedMeshType = (renderMode == RenderMode.PointCloud) ? 
                    MeshingSubsystemComponent.MeshType.PointCloud : 
                    MeshingSubsystemComponent.MeshType.Triangles;

                switch (renderMode)
                {
                    case RenderMode.None:
                        break;
                    case RenderMode.Wireframe:
                        _meshingSubsystemComponent.PrefabRenderer.sharedMaterial = _wireframeMaterial;
                        break;
                    case RenderMode.Colored:
                        _meshingSubsystemComponent.PrefabRenderer.sharedMaterial = _coloredMaterial;
                        break;
                    case RenderMode.PointCloud:
                        _meshingSubsystemComponent.PrefabRenderer.sharedMaterial = _pointCloudMaterial;
                        break;
                    case RenderMode.Occlusion:
                        _meshingSubsystemComponent.PrefabRenderer.sharedMaterial = _occlusionMaterial;
                        break;
                    default:
                        throw new ArgumentOutOfRangeException($"unknown renderMode value: {renderMode}");
                }
                

                _meshingSubsystemComponent.DestroyAllMeshes();
                _meshingSubsystemComponent.RefreshAllMeshes();
            }
            Debug.Log("Rendered Set.");
        }
        void Update(){
            Debug.Log("Update mesh.");
            _meshingSubsystemComponent.gameObject.transform.position = _camera.gameObject.transform.position;
        }

        string Vector3ArrayToJsonString(Vector3[] array)
        {
            // Serialize the Vector3 array
            JsonArray serializedArray = new JsonArray();
            foreach (Vector3 vector in array)
            {
                JsonArray serializedVector = new JsonArray();
                serializedVector.Add(vector.x);
                serializedVector.Add(vector.y);
                serializedVector.Add(vector.z);
                serializedArray.Add(serializedVector);
            }

            string jsonString = serializedArray.ToString();;
            return jsonString;
        }

        string Vector2ArrayToJsonString(Vector2[] array)
        {
            // Serialize the Vector3 array
            JsonArray serializedArray = new JsonArray();
            foreach (Vector2 vector in array)
            {
                JsonArray serializedVector = new JsonArray();
                serializedVector.Add(vector.x);
                serializedVector.Add(vector.y);
                serializedArray.Add(serializedVector);
            }

            string jsonString = serializedArray.ToString();;
            return jsonString;
        }

        /// <summary>
        /// Handles the MeshReady event, which tracks and assigns the correct mesh renderer materials.
        /// </summary>
        /// <param name="meshId">Id of the mesh that got added / upated.</param>
        private void HandleOnMeshReady(UnityEngine.XR.MeshId meshId)
        {
            if (_meshingSubsystemComponent.meshIdToGameObjectMap.TryGetValue(meshId, out var meshGameObject))
            {
                meshGameObject.GetComponent<Renderer>().enabled = false;
                Debug.Log("here");
                var mf = meshGameObject.GetComponent<MeshFilter>();
                MeshCollider meshCollider = meshGameObject.AddComponent<MeshCollider>();
                meshCollider.sharedMesh = mf.mesh;
            }
        }
    }
}