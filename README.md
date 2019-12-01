# Facelandmark
Project Face landmark

# function.py 
เป็น source code ที่ใช้ในการสร้าง function 
# home.py 
เป็น source code ที่ใช้ในการเรียกใช้ฟังก์ชั้น และสามารถก้ไขข้อมูลต่างๆได้โดบยไม่ต้องเข้าไป
แก้ไขข้อมูลใน file function.py โดยจะส่ง parameters เข้าไป

# -function readimage
เป็น function ที่ใช้สำหรับ นำรูปภาพขึ้นมาแสดงสามารถเลือกได้ว่าจะนำรูปภาพที่มี 
landmark หรือไม่มี landmark ขึ้นมาแสดงได้
#  -> Parameters ที่ส่งเข้าไปคือ
    1. p            -- ใช้การเก็บชื่อ file .dat ทีใช้ในการ predictor 
                    **ตัวอย่าง -> p = "face_landmarks_18-11-62.dat";
    2. detector     -- ใช้ในการเก็บคำสั่ง detection
                    **ตัวอย่าง -> detector = dlib.get_frontal_face_detector();
    3. predictor    -- ใช้ในการเก็บคำสั่ง predictor
                    **ตัวอย่าง -> predictor = dlib.shape_predictor(p);
    4. width        -- ใช้ในการเก็บขนาดของหน้าจอแสดงผล (ใช้ในการ resize)
                    **ตัวอย่าง -> width = 500;
    5. color        -- ใช้ในการเก็บข้อมูลของสี
                    **ตัวอย่าง -> color = cv2.COLOR_BGR2GRAY;
    6. msshow       -- ใช้ในการเก็บข้อความ heading ที่ใช้ในการแสดงผล
                    **ตัวอย่าง -> msshow = "output"
    7. face         -- ใช้ในการเก็บข้อความที่จะแสดงเมื่อมีการ detect ที่ใบหน้า
                    **ตัวอย่าง -> face = "face #{}"
    8. mschoose     -- ใช้ในการเก็บข้อความที่จะแสดงเพื่อให้ผู้ใช้เลือก
                    **ตัวอย่าง -> mschoose = "please choose d|l: "
    9. ir           -- ใช้ในการเก็บชื่อ file ที่ต้องการอ่าน (imread)
                    **ตัวอย่าง -> ir = "test/te1.jpg"
    10.keyd         -- ใช้ในการเก็บปุ่มกดที่ต้องการแสดงภาพ เมื่อกดปุ่ม d จะทำการแสดงภาพที่ไม่มี landmark
                    **ตัวอย่าง -> keyd = ('d')
    11.keyl         -- ใช้ในการเก็บปุ่มกดที่ต้องการแสดงภาพ เมื่อกดปุ่ม l จะทำการแสดงภาพที่มี landmark
                    **ตัวอย่าง -> keyl = ('l') 

# -function showlandmark
เป็น function ที่ใช้สำหรับนำจุด landmark มาแสดงบนใบหน้าแล้วทำการ save เป็นfile .jpg
#  -> Parameters ที่ส่งเข้าไปคือ
    1. p            -- ใช้การเก็บชื่อ file .dat ทีใช้ในการ predictor 
                    **ตัวอย่าง -> p = "face_landmarks_18-11-62.dat";
    2. detector     -- ใช้ในการเก็บคำสั่ง detection
                    **ตัวอย่าง -> detector = dlib.get_frontal_face_detector();
    3. predictor    -- ใช้ในการเก็บคำสั่ง predictor
                    **ตัวอย่าง -> predictor = dlib.shape_predictor(p);
    4. width        -- ใช้ในการเก็บขนาดของหน้าจอแสดงผล (ใช้ในการ resize)
                    **ตัวอย่าง -> width = 500;
    5. color        -- ใช้ในการเก็บข้อมูลของสี
                    **ตัวอย่าง -> color = cv2.COLOR_BGR2GRAY;
    6. keyq         -- ใช้ในการเก็บปุ่มกดที่ต้องการออกจากการแสดงผล เมื่อกดปุ่ม q จะออกจากการแสดงผล
                    **ตัวอย่าง -> keyq = ('q')
    7. keyl         -- ใช้ในการเก็บปุ่มกดที่ต้องการ save รูปภาพ เมื่อกดปุ่ม l จะทำการ save รูปภาพที่มี landmark
                    **ตัวอย่าง -> keyl = ('l') 
    8. extension    -- ใช้ในการเก็บชื่อนามสกุล file
                    **ตัวอย่าง -> extension = ".jpg"
    9. msshow       -- ใช้ในการเก็บข้อความ heading ที่ใช้ในการแสดงผล
                    **ตัวอย่าง -> msshow = "output"
    10.id           -- ใช้ในการเก็บตัวเลขเริ่มต้นเพื่อวนลูบ เพื่อ save ภาพ
                    **ตัวอย่าง -> id = 0
    11.ir           -- ใช้ในการเก็บชื่อ file ที่ต้องการอ่าน (imread)
                    **ตัวอย่าง -> ir = "picture/train/t5.jpg"
    12.iwl          -- ใช้สำหรับเก็บ path file ที่ใช้ในการ save รูปภาพที่มี landmark
                    **ตัวอย่าง -> iwl = "picture/savelandmark/la."

# -function detectface 
เป็น function ทีใช้สำหรับ detect ใบหน้าผ่านกล้องรวมไปถึงสามารถ กด key เพื่อบันทึกรูปภาพ สามารถที่จะเลือก
ได้ว่าจะบันทึกรูปแบบ ที่มี landmark หรือ ไม่มี landmark รูปภาพที่บันทึกจะอยู่ในรูปแบบ file .jpg
#  -> Parameters ที่ส่งเข้าไปคือ
    1. p            -- ใช้การเก็บชื่อ file .dat ทีใช้ในการ predictor 
                    **ตัวอย่าง -> p = "face_landmarks_18-11-62.dat";
    2. detector     -- ใช้ในการเก็บคำสั่ง detection
                    **ตัวอย่าง -> detector = dlib.get_frontal_face_detector();
    3. predictor    -- ใช้ในการเก็บคำสั่ง predictor
                    **ตัวอย่าง -> predictor = dlib.shape_predictor(p);
    4. color        -- ใช้ในการเก็บข้อมูลของสี
                    **ตัวอย่าง -> color = cv2.COLOR_BGR2GRAY;
    5. keyq         -- ใช้ในการเก็บปุ่มกดที่ต้องการออกจากการแสดงผล เมื่อกดปุ่ม q จะออกจากการแสดงผล
                    **ตัวอย่าง -> keyq = ('q')
    6. keyl         -- ใช้ในการเก็บปุ่มกดที่ต้องการ save รูปภาพ เมื่อกดปุ่ม l จะทำการ save รูปภาพที่มี landmark
                    **ตัวอย่าง -> keyl = ('l')
    7. extension    -- ใช้ในการเก็บชื่อนามสกุล file
                    **ตัวอย่าง -> extension = ".jpg"
    8. face         -- ใช้ในการเก็บข้อความที่จะแสดงเมื่อมีการ detect ที่ใบหน้า
                    **ตัวอย่าง -> face = "face #{}"
    9. msshow       -- ใช้ในการเก็บข้อความ heading ที่ใช้ในการแสดงผล
                    **ตัวอย่าง -> msshow = "output"
    10.keyd        -- ใช้ในการเก็บปุ่มกดที่ต้องการ save รูปภาพ เมื่อกดปุ่ม d จะทำการ save รูปภาพที่ไม่มี landmark
                    **ตัวอย่าง -> keyl = ('d')
    11.read        -- ใช้ในการเก็บข้อมูลที่ต้องการอ่านเช่น ให้อ่านจากวิดีโอ หรืออ่านจากกล้อง
                    **ตัวอย่าง -> read = cv2.VideoCapture(0)
    12.id          -- ใช้ในการเก็บตัวเลขเริ่มต้นเพื่อวนลูบ เพื่อ save ภาพ
                    **ตัวอย่าง -> id = 1
    13.iwd         -- ใช้สำหรับเก็บ path file ที่ใช้ในการ save รูปภาพที่ไม่มี landmark
                    **ตัวอย่าง -> iwd = "picture/detection/de."
    14.iwl         -- ใช้สำหรับเก็บ path file ที่ใช้ในการ save รูปภาพที่มี landmark
                    **ตัวอย่าง ->  iwl = "picture/landmark/la."

# -function checkerror
เป็น function ที่ใช้สำหรับ check error ของโมเดล แล้วนำมาเปรียบเทียบ
#  -> Parameters ที่ส่งเข้าไปคือ
    1. mserror      -- ใช้ในการเก็บข้อความที่ใช้ในการแสดงผล error
                    **ตัวอย่าง -> mserror = "error of the model: {} is {}"
    2. repart       -- ใช้ในการเก็บข้อมูล REG_PART
                    **ตัวอย่าง -> repart = "part name='[0-9]+'"
    3. renum        -- ใช้ในการเก็บข้อมูล REG_NUM
                    **ตัวอย่าง -> renum = "[0-9]+"
    4. dir          -- ใช้ในการเก็บข้อมูล dataset path (ibug_dir)
                    **ตัวอย่าง -> dir = "face_landmark"
    5. newmodel     -- ใช้ในการเก็บชื่อ file .dat ของ new model
                    **ตัวอย่าง ->  newmodel = "face_landmarks_18-11-62.dat"
    6. dlibmodel    -- ใช้ในการเก็บชื่อ file .dat ของ dlib model
                    **ตัวอย่าง -> dlibmodel = "face_landmarks_68_30.dat"
    7. filexml      -- ใช้ในการเก็บชื่อ file .xml train&test model
                    **ตัวอย่าง -> filexml = f"face_landmarks_18-11-62_test.xml"

# -function replacevalue
เป็น function ที่ใช้ในการ เปลี่ยนค่าตัวเลขใน file trainimage .xml ให้เป็นเลขสองหลักเพื่อใช้ในการ train model
 #  -> Parameters ที่ส่งเข้าไปคือ
    1. dir          -- ใช้ในการเก็บข้อมูล dataset path (my_dir)
                    **ตัวอย่าง -> 'C://Users//FRONTIS//OneDrive - Frontis//face_landmark//filereplace'
    2. rpwhat       -- ใช้ในการเก็บข้อมูลทีต้องการ replace ว่าต้องการที่จะ replace อะไร
                    **ตัวอย่าง -> rpwhat = "name='0'"
    3. rpwith       -- ใช้ในการเก็บข้อมูลทีต้องการ replace ว่าต้องการที่จะ replace เป็นอะไร
                    **ตัวอย่าง -> rpwith = "name='00'"   
    4. readpath    -- ใช้ในการเก็บข้อมูล ประเภทที่ต้องการอ่าน file  
                    **ตัวอย่าง -> readpath = 'r+'

# -function trainmodel
เป็น function ที่ใช้ในการ train model
 #  -> Parameters ที่ส่งเข้าไปคือ
    1. mserror      -- ใช้ในการเก็บข้อความที่ใช้ในการแสดงผล error
                    **ตัวอย่าง -> mserror = "error of the model: {} is {}"
    2. repart       -- ใช้ในการเก็บข้อมูล REG_PART
                    **ตัวอย่าง -> repart = "part name='[0-9]+'"
    3. renum        -- ใช้ในการเก็บข้อมูล REG_NUM
                    **ตัวอย่าง -> renum = "[0-9]+"
    4. eye          -- ใช้ในการเก็บข้อมูลจุด landmark ของคิ้วและดวงตา (EYE_EYEBROWS)
                    **ตัวอย่าง -> eye = [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 36,
                                      37, 38, 39,40, 41, 42, 43, 44, 45, 46, 47]
    5. nose         -- ใช้ในการเก็บข้อมูลจุด landmark ของจมูกและปาก (NOSE_MOUTH)
                    **ตัวอย่าง -> nose = [27, 28, 29, 30, 31, 32, 33, 34, 35, 48, 49,
                                       50, 51, 52,53, 54, 55, 56, 57, 58, 59, 60,
                                       61, 62, 63, 64, 65, 66, 67]
    6. face         -- ใช้ในการเก็บข้อมูลจุด landmark ของใบหน้า (FACE_CONTOUR)
                    **ตัวอย่าง -> face = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                                       14, 15, 16]
    7. landmark     -- ใช้ในการเก็บข้อมูลจุดทั้งหมดบนใบหน้า (ALL_LANDMARKS)
                    **ตัวอย่าง -> landmark = range(0, 68)
    8. train        -- ใช้ในการเก็บชื่อ file .xml ที่ใช้ในการ train model
                    **ตัวอย่าง -> train = "labels_ibug_300W_train.xml"
    9. test         -- ใช้ในการเก็บชื่อ file .xml ที่ใช้ในการ test model
                    **ตัวอย่าง -> test = "face_landmarks_test.xml"
    10.opt          -- ใช้ในการเก็บข้อมูล (option)
                    **ตัวอย่าง -> opt = dlib.shape_predictor_training_options()
    11.tree         -- ใช้ในการเก็บข้อมูล (options.tree_depth)
                    **ตัวอย่าง -> tree = 3
    12.nu           -- ใช้ในการเก็บข้อมูล (options.nu)
                    **ตัวอย่าง -> nu = 0.1
    13.cascade      -- ใช้ในการเก็บข้อมูล (options.cascade_depth)
                    **ตัวอย่าง -> cascade = 10
    14.feature      -- ใช้ในการเก็บข้อมูล (options.feature_pool_size)
                    **ตัวอย่าง -> feature = 150
    15.ntest        -- ใช้ในการเก็บข้อมูล (options.num_test_splits)
                    **ตัวอย่าง -> ntest = 350
    16.osamount     -- ใช้ในการเก็บข้อมูล (options.oversampling_amount)
                    **ตัวอย่าง -> osamount = 5
    17.ostranslation   -- ใช้ในการเก็บข้อมูล (options.oversampling_translation_jitter)
                    **ตัวอย่าง ->  ostranslation = 0
    18.be           -- ใช้ในการเก็บข้อมูล (options.be_verbose)
                    **ตัวอย่าง -> be = True
    19.nthreads     -- ใช้ในการเก็บข้อมูล (options.num_threads)
                    **ตัวอย่าง -> nthreads = 1
    20.namemodel    -- ใช้ในการเก็บข้อมูลชื่อ model (model name)
                    **ตัวอย่าง -> namemodel = "face_landmarks_68"

# -function imagejitter
เป็น function ที่ใช้ในการ generate รูปภาพ ให้ออกมามีความหลากหลายไม่ซ้ำกัน
  #  -> Parameters ที่ส่งเข้าไปคือ
    1. default      -- ใช้ในการเก็บข้อมูลค่าเริ่มต้นของจำนวนรูปภาพที่ต้องการ generate
                    **ตัวอย่าง -> default = 100
    2. msload       -- ใช้ในการเก็บข้อมูลข้อความที่จะแสดงตอน load image
                    **ตัวอย่าง -> msload = "[INFO] loading example image..."
    3. rotation     -- ใช้ในการเก็บข้อมูล (rotation_range)
                    **ตัวอย่าง -> rotation = 0.2
    4. zoom         -- ใช้ในการเก็บข้อมูล (zoom_range)
                    **ตัวอย่าง -> zoom = 0.10
    5. width        -- ใช้ในการเก็บข้อมูล (width_shift_range)
                    **ตัวอย่าง -> width = 0.1
    6. height       -- ใช้ในการเก็บข้อมูล (height_shift_range)
                    **ตัวอย่าง -> height = 0.1
    7. shear        -- ใช้ในการเก็บข้อมูล (shear_range)
                    **ตัวอย่าง -> shear = 0.10
    8. status       -- ใช้ในการเก็บข้อมูล (horizontal_flip)
                    **ตัวอย่าง -> status = True
    9. mode         -- ใช้ในการเก็บข้อมูล (fill_mode)
                    **ตัวอย่าง -> mode = "nearest"
    10.total        -- ใช้ในการเก็บข้อมูล ตัวเลขเพื่อใช้ในการวนลูบ (total)
                    **ตัวอย่าง -> total = 0
    11.msgenerat    -- ใช้ในการเก็บข้อมูลข้อความที่จะแสดงตอน generator
                    **ตัวอย่าง -> msgenerat = "[INFO] generating images..."
    12.file         -- ใช้ในการเก็บนามสกุล file ที่ต้องการบันทึรูปภาพ
                    **ตัวอย่าง -> file = "jpg"
