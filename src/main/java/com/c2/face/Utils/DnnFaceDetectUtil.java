package com.c2.face.Utils;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.opencv.opencv_java;
import org.opencv.core.*;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.File;
import java.io.InputStream;
import java.nio.file.Files;

/**
 * 基于 OpenCV DNN 的人脸检测工具类
 * 支持置信度阈值调节，可减少动漫脸误报
 */
public class DnnFaceDetectUtil {

    static {
        // 自动加载 OpenCV native
        Loader.load(opencv_java.class);
    }

    private static final Net FACE_NET;

    static {
        try {
            File proto = copyResource("dnn/deploy.prototxt");
            File model = copyResource("dnn/res10_300x300_ssd_iter_140000.caffemodel");

            FACE_NET = Dnn.readNetFromCaffe(proto.getAbsolutePath(), model.getAbsolutePath());
            if (FACE_NET.empty()) {
                throw new RuntimeException("DNN 人脸模型加载失败，net 为空");
            }

            System.out.println("✅ DNN 人脸模型加载成功");

        } catch (Exception e) {
            throw new RuntimeException("DNN 人脸模型加载失败", e);
        }
    }

    /**
     * 判断图片是否包含人脸
     * @param imageFile 图片文件
     * @param confidenceThreshold 置信度阈值，0~1，默认可取 0.6~0.7
     * @return true:检测到人脸, false:未检测到
     */
    public static boolean hasHumanFace(File imageFile, double confidenceThreshold) {
        Mat image = Imgcodecs.imread(imageFile.getAbsolutePath());
        if (image.empty()) return false;

        // 预处理 blob
        Mat blob = Dnn.blobFromImage(
                image,
                1.0,
                new Size(300, 300),
                new Scalar(104, 177, 123), // mean 值
                false,
                false
        );

        FACE_NET.setInput(blob);

        // forward
        Mat detections = FACE_NET.forward();

        // reshape 为 [N,7] 每行：[batch_id, class_id, confidence, x1, y1, x2, y2]
        detections = detections.reshape(1, (int) detections.total() / 7);

        for (int i = 0; i < detections.rows(); i++) {
            double confidence = detections.get(i, 2)[0];
            if (confidence >= confidenceThreshold) {
                return true;
            }
        }

        return false;
    }

    /** 重载，默认置信度 0.6 */
    public static boolean hasHumanFace(File imageFile) {
        return hasHumanFace(imageFile, 0.6);
    }

    /** 从 classpath 拷贝模型到临时文件 */
    private static File copyResource(String path) throws Exception {
        InputStream is = DnnFaceDetectUtil.class
                .getClassLoader()
                .getResourceAsStream(path);

        if (is == null) {
            throw new RuntimeException("模型文件不存在: " + path);
        }

        File temp = File.createTempFile("dnn_", path.substring(path.lastIndexOf(".")));
        Files.copy(is, temp.toPath(), java.nio.file.StandardCopyOption.REPLACE_EXISTING);
        temp.deleteOnExit();
        return temp;
    }
}
