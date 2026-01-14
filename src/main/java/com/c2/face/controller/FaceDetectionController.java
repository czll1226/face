package com.c2.face.controller;

import com.c2.face.Utils.DnnFaceDetectUtil;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;

@CrossOrigin(origins = "*") // 添加这一行，允许所有来源访问
@RestController
@RequestMapping("/face")
public class FaceDetectionController {

    @PostMapping("/detect")
    public boolean detect(@RequestParam("file") MultipartFile file) throws Exception {

        File temp = File.createTempFile("upload_", ".jpg");
        file.transferTo(temp);

        long l = System.currentTimeMillis();
        boolean result = DnnFaceDetectUtil.hasHumanFace(temp,0.99f);
        long l2 = System.currentTimeMillis() - l;
        System.out.println("======================time:"+l2);
        System.out.println("检测结果: "+result);
        temp.delete();
        return result;
    }
}
