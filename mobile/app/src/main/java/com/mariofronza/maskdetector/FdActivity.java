package com.mariofronza.maskdetector;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.util.Collections;
import java.util.List;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraActivity;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;

import android.content.Context;
import android.os.Bundle;
import android.util.Log;
import android.view.WindowManager;
import android.widget.TextView;

public class FdActivity extends CameraActivity implements CvCameraViewListener2 {

    private Mat mRgba;
    private Mat mGray;
    private CascadeClassifier mJavaDetector;
    private CameraBridgeViewBase mOpenCvCameraView;
    private TextView usingMask;
    private Mat CNN_input;

    private ImageProcessor imageProcessor;
    private TensorImage tImage;
    private MappedByteBuffer model;

    private static final int DIM_BATCH_SIZE = 1;
    private static final int DIM_PIXEL_SIZE = 1;
    private static final int DIM_HEIGHT = 224;
    private static final int DIM_WIDTH = 672;
    private static final int BYTES = 4;

    private static float prob = 0.0f;
    private float[][] ProbArray = null;
    private String quantity;
    private TextView quantityText;

    private ByteBuffer imgData = null;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            if (status == LoaderCallbackInterface.SUCCESS) {
                System.loadLibrary("detection_based_tracker");

                try {
                    InputStream is = getResources().openRawResource(R.raw.lbpcascade_frontalface);
                    File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
                    File mCascadeFile = new File(cascadeDir, "lbpcascade_frontalface.xml");
                    File mouthCascadeFile = new File(cascadeDir, "haarcascade_mcs_mouth.xml");
                    FileOutputStream os = new FileOutputStream(mCascadeFile);
                    FileOutputStream osMouth = new FileOutputStream(mouthCascadeFile);

                    byte[] buffer = new byte[4096];
                    int bytesRead;
                    while ((bytesRead = is.read(buffer)) != -1) {
                        os.write(buffer, 0, bytesRead);
                        osMouth.write(buffer, 0, bytesRead);
                    }
                    is.close();
                    os.close();
                    osMouth.close();

                    mJavaDetector = new CascadeClassifier(mCascadeFile.getAbsolutePath());
                    if (mJavaDetector.empty()) {
                        mJavaDetector = null;
                    }

                    cascadeDir.delete();

                } catch (IOException e) {
                    e.printStackTrace();
                }
                mOpenCvCameraView.setCameraIndex(1);
                mOpenCvCameraView.setMinimumWidth(400);
                mOpenCvCameraView.enableView();


            } else {
                super.onManagerConnected(status);
            }
        }
    };

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.face_detect_surface_view);
        quantityText = findViewById(R.id.numberText);
        mOpenCvCameraView = findViewById(R.id.fd_activity_surface_view);
        mOpenCvCameraView.setVisibility(CameraBridgeViewBase.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
        usingMask = findViewById(R.id.tvMask);
        imageProcessor = new ImageProcessor.Builder()
                .add(new ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
                .build();
        quantity = getIntent().getStringExtra("quantity");
        quantityText.setText(quantity);
        tImage = new TensorImage(DataType.UINT8);
        ProbArray = new float[1][2];
        imgData = ByteBuffer.allocateDirect(DIM_BATCH_SIZE * DIM_HEIGHT * DIM_WIDTH * DIM_PIXEL_SIZE * BYTES);
        imgData.order(ByteOrder.nativeOrder());
        setViewText("");
    }

    @Override
    protected List<? extends CameraBridgeViewBase> getCameraViewList() {
        return Collections.singletonList(mOpenCvCameraView);
    }

    public void setViewText(final String text) {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                usingMask.setText(text);
            }
        });
    }

    public void onCameraViewStarted(int width, int height) {
        mGray = new Mat();
        mRgba = new Mat();
        CNN_input = new Mat();
    }

    public void onCameraViewStopped() {
        mGray.release();
        mRgba.release();
    }

    public void onDestroy() {
        super.onDestroy();
        mOpenCvCameraView.disableView();
    }

    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();

        MatOfRect faces = new MatOfRect();

        if (mJavaDetector != null) {
            mJavaDetector.detectMultiScale(mGray, faces, 1.1, 2, 2,
                    new Size(450, 450), new Size());
        }

        Rect[] facesArray = faces.toArray();

        if (facesArray.length == 0) {
            setViewText("");
        }

        for (Rect rect : facesArray) {
            Imgproc.rectangle(mRgba, rect.tl(), rect.br(), new Scalar(255, 0, 0), 3);
            setViewText("Coloque a máscara");
            Imgproc.resize(mGray, CNN_input, new org.opencv.core.Size(rect.width, rect.height));
//            detectMask(CNN_input);
        }
        return mRgba;
    }

    private void detectMask(Mat input) {
        try {
            model = FileUtil.loadMappedFile(this, "model.tflite");
            Interpreter tflite = new Interpreter(model);
            convertMattoTfLiteInput(input);
            tflite.run(imgData, ProbArray);
            Log.e("FD", "Prob: " + maxProbIndex(ProbArray[0]));
            if (ProbArray[0][0] > ProbArray[0][1]) {
                Log.e("FD", "sem máscara");
            } else {
                Log.e("FD", "com máscara");
            }
//            Log.e("FD", "Prob1: " + ProbArray[0][0]);
//            Log.e("FD", "Prob2: " + ProbArray[0][1]);
        } catch (IOException e) {
            Log.e("tfliteSupport", "Error reading model", e);
        }
    }

    private int maxProbIndex(float[] probs) {
        int maxIndex = -1;
        float maxProb = 0.0f;
        for (int i = 0; i < probs.length; i++) {
            if (probs[i] > maxProb) {
                maxProb = probs[i];
                maxIndex = i;
            }
        }
        prob = maxProb;
        return maxIndex;
    }

    private void convertMattoTfLiteInput(Mat mat) {

        imgData.rewind();
        for (int i = 0; i < 28; ++i) {
            for (int j = 0; j < 28; ++j) {
                imgData.putFloat((float) mat.get(i, j)[0]);
            }
        }
    }

    @Override
    public void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    @Override
    public void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null) {
            mOpenCvCameraView.disableView();
        }
    }
}
