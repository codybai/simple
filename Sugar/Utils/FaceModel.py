import numpy as np
class FaceModel():
    def __init__(self,facePoint_):
        self.LeftEye = [
            [-0.425919, 0.440593, 0.224658],#1
            [- 0.370646, 0.481375, 0.283101],#2
            [- 0.293811, 0.494563, 0.307038],#3
            [- 0.225717, 0.478080, 0.304148],#4
            [- 0.172256, 0.424206, 0.293831],#5
            [- 0.245222, 0.408030, 0.311554],#6
            [- 0.307160, 0.396866, 0.305652],#7
            [- 0.368465, 0.409455, 0.279033],#8
        ]
        self.RightEye = [
            [0.172256, 0.424206, 0.293831],#5
            [0.225717, 0.478080, 0.304148],#4
            [0.293811, 0.494563, 0.307038],#3
            [0.370646, 0.481375, 0.283101],#2
            [0.425919, 0.440593, 0.224658],#1
            [0.368465, 0.409455, 0.279033],#8
            [0.307160, 0.396866, 0.305652],#7
            [0.245222, 0.408030, 0.311554],#6
        ]
        self.Nose = [
            [-0.001642, 0.443398, 0.407059],
            [0.000134, 0.365501, 0.459487],
            [0.001943, 0.288615, 0.519069],
            [0.003661, 0.140981, 0.622702],
            [- 0.092302, 0.407880, 0.356290],
            [- 0.147194, 0.150407, 0.409337],
            [- 0.178635, 0.082301, 0.376230],
            [- 0.122133, 0.020136, 0.434100],
            [- 0.101430, 0.056324, 0.461743],
            [0.001467, 0.016863, 0.506306],
            [0.104120, 0.054009, 0.461953],
            [0.126418, 0.018170, 0.430352],
            [0.178313, 0.081167, 0.375544],
            [0.146113, 0.149723, 0.405833],
            [0.082965, 0.408309, 0.358893],
        ]
        self.Mouse = [
            [-0.212068, - 0.180958, 0.378429],
            [- 0.127431, - 0.128284, 0.475622],
            [- 0.053701, - 0.104428, 0.516951],
            [0.003923, - 0.125636, 0.523327],
            [0.068744, - 0.106478, 0.514485],
            [0.150545, - 0.130112, 0.466677],
            [0.221960, - 0.177851, 0.379463],
            [0.164533, - 0.224281, 0.417805],
            [0.086272, - 0.285726, 0.473047],
            [- 0.006493, - 0.290790, 0.488832],
            [- 0.090219, - 0.279947, 0.470324],
            [- 0.157928, - 0.237369, 0.417759],
            [- 0.188991, - 0.180038, 0.383582],
            [- 0.086879, - 0.165678, 0.435284],
            [- 0.002880, - 0.172800, 0.453929],
            [0.076486, - 0.180413, 0.436466],
            [0.206915, - 0.181761, 0.384100],
            [0.085111, - 0.185016, 0.445173],
            [0.008764, - 0.185477, 0.442922],
            [- 0.098632, - 0.183769, 0.412114],
        ]
        self.Face = [
            [-0.656176, 0.441354, - 0.216053],
            [- 0.641410, 0.289355, - 0.266398],
            [- 0.599689, 0.127149, - 0.254836],
            [- 0.566327, - 0.028363, - 0.202199],
            [- 0.530780, - 0.208015, - 0.141956],
            [- 0.423474, - 0.376734, 0.001086],
            [- 0.202589, - 0.532048, 0.198746],
            [0.003592, - 0.570074, 0.287670],
            [0.194170, - 0.542960, 0.193354],
            [0.439780, - 0.365412, - 0.011981],
            [0.550214, - 0.179881, - 0.215860],
            [0.571561, - 0.036432, - 0.261662],
            [0.595106, 0.105059, - 0.286407],
            [0.629346, 0.272084, - 0.286301],
            [0.641330, 0.425274, - 0.235518],
        ]
        self.LeftBrow =[
            [-0.534252, 0.651108, 0.134591],
            [- 0.461283, 0.714408, 0.208611],
            [- 0.405465, 0.720242, 0.268854],
            [- 0.342518, 0.718276, 0.311343],
            [- 0.265680, 0.704897, 0.349782],
            [- 0.252757, 0.652961, 0.358036],
            [- 0.319742, 0.668340, 0.332218],
            [- 0.398500, 0.674881, 0.293097],
            [- 0.464661, 0.670330, 0.236675],
        ]
        self.RightBrow = [
            [0.251438, 0.716975, 0.350636],
            [0.311522, 0.726471, 0.318951],
            [0.376474, 0.727576, 0.270879],
            [0.432966, 0.722624, 0.218324],
            [0.510718, 0.659094, 0.133847],
            [0.436023, 0.678895, 0.245483],
            [0.370447, 0.683969, 0.296816],
            [0.301848, 0.678538, 0.333791],
            [0.242651, 0.667189, 0.359632],
        ]
        self.PP_points = self.LeftEye+self.RightEye+\
                      self.Nose+\
                      self.Mouse+\
                      self.Face+\
                      self.LeftBrow+self.RightBrow

        self.lefteye = [
            [-0.643667, 0.222751, 0.225499],

            [- 0.567720, 0.362798, 0.310201],
            [- 0.401194, 0.404963, 0.389812],
            [- 0.266332, 0.328440, 0.378106],

            [- 0.212005, 0.215182, 0.336718],
            [- 0.274377, 0.168061, 0.346075],
            [- 0.351221, 0.140792, 0.351298],
            [- 0.477728, 0.138626, 0.326400],
        ]
        self.righteye = [
            [0.212005, 0.215182, 0.336718],

            [0.266332, 0.328440, 0.378106],
            [0.401194, 0.404963, 0.389812],
            [0.567720, 0.362798, 0.310201],

            [0.643667, 0.222751, 0.225499],
            [0.477726, 0.138624, 0.326400],
            [0.351221, 0.140792, 0.351296],
            [0.274377, 0.168061, 0.346075],
        ]
        self.nose = [
            [0.000000, 0.249221, 0.506346],
            [- 0.000020, 0.122646, 0.524472],
            [- 0.000057, 0.035274, 0.570447],
            [- 0.000004, - 0.079987, 0.661684],
            [- 0.103037, 0.200417, 0.454122],
            [- 0.108885, - 0.054141, 0.505723],
            [- 0.151901, - 0.089255, 0.481830],
            [- 0.089883, - 0.155296, 0.516148],
            [- 0.051698, - 0.133247, 0.555747],
            [0.000001, - 0.178858, 0.553901],
            [0.051681, - 0.133247, 0.555751],
            [0.089835, - 0.155298, 0.516150],
            [0.151627, - 0.089281, 0.481833],
            [0.108605, - 0.054149, 0.505728],
            [0.101480, 0.200679, 0.454121],
        ]
        self.mouse = [
            [-0.193654, - 0.298529, 0.437177],
            [- 0.100081, - 0.283034, 0.518146],
            [- 0.045665, - 0.271066, 0.549374],
            [- 0.000001, - 0.290789, 0.550152],
            [0.045663, - 0.271065, 0.549372],
            [0.100037, - 0.283032, 0.518141],
            [0.193411, - 0.298518, 0.437160],
            [0.127189, - 0.336053, 0.479596],
            [0.075928, - 0.365846, 0.506781],
            [- 0.000002, - 0.370111, 0.516977],
            [- 0.075952, - 0.365848, 0.506786],
            [- 0.133483, - 0.343931, 0.476519],
            [- 0.169031, - 0.299944, 0.425218],
            [- 0.060590, - 0.303100, 0.445966],
            [- 0.004352, - 0.297862, 0.455051],
            [0.020694, - 0.298573, 0.453631],
            [0.177920, - 0.301072, 0.430174],
            [0.044747, - 0.302323, 0.446415],
            [- 0.000002, - 0.297907, 0.449052],
            [- 0.067832, - 0.303965, 0.441931],
        ]
        self.face = [
            [-0.744524, 0.370140, - 0.114816],
            [- 0.741091, 0.204283, - 0.188177],
            [- 0.698763, 0.028032, - 0.222621],
            [- 0.641970, - 0.119308, - 0.190447],
            [- 0.539659, - 0.293869, - 0.138438],
            [- 0.393605, - 0.442794, 0.032684],
            [- 0.178753, - 0.575119, 0.200795],
            [0.000000, - 0.640921, 0.325525],
            [0.178832, - 0.575092, 0.200794],
            [0.393941, - 0.442632, 0.032684],
            [0.539858, - 0.293721, - 0.138438],
            [0.642024, - 0.119255, - 0.190447],
            [0.698773, 0.028035, - 0.222621],
            [0.741107, 0.204336, - 0.188177],
            [0.744639, 0.370537, - 0.114816],
        ]

        self.leftbrow = [
            [-0.534252, 0.651108, 0.134591],
            [- 0.461283, 0.714408, 0.208611],
            [- 0.405465, 0.720242, 0.268854],
            [- 0.342518, 0.718276, 0.311343],
            [- 0.265680, 0.704897, 0.349782],
            [- 0.252757, 0.652961, 0.358036],
            [- 0.319742, 0.668340, 0.332218],
            [- 0.398500, 0.674881, 0.293097],
            [- 0.464661, 0.670330, 0.236675],
        ]
        self.rightbrow = [
            [0.251438, 0.716975, 0.350636],
            [0.311522, 0.726471, 0.318951],
            [0.376474, 0.727576, 0.270879],
            [0.432966, 0.722624, 0.218324],
            [0.510718, 0.659094, 0.133847],
            [0.436023, 0.678895, 0.245483],
            [0.370447, 0.683969, 0.296816],
            [0.301848, 0.678538, 0.333791],
            [0.242651, 0.667189, 0.359632]
        ]

        self.MK_points = self.lefteye+self.righteye+\
                           self.nose+\
                           self.mouse+\
                           self.face+\
                           self.leftbrow+self.rightbrow
        ex = np.zeros(facePoint_.shape)[:,0:1]
        facePoint = np.concatenate([facePoint_, ex], 1)

        facePoint = list(facePoint)
        self.u_lefteye = facePoint[51:59]
        self.u_righteye = facePoint[61:69]
        self.u_nose = facePoint[71:86]
        self.u_mouse = facePoint[86:106]
        self.u_face = facePoint[2:32:2]
        self.u_leftbrow = facePoint[33:42]
        self.u_rightbrow = facePoint[42:51]

        self.US_points = self.u_lefteye+self.u_righteye+\
                           self.u_nose+\
                           self.u_mouse+\
                           self.u_face+\
                           self.u_leftbrow+self.u_rightbrow

        self.dict = {
            'LE':0,
            'RE':8,
            'N': 16,
            'M': 31,
            'F': 51,
            'LB':66,
            'RB':75,
            'E':0,
            'B':66,
        }
        self.dict_len = {
            'LE': 8,
            'RE': 8,
            'N': 15,
            'M': 20,
            'F': 15,
            'LB': 9,
            'RB': 9,
            'E':16,
            'B':18
        }
        self.dict_usr = {
            'PP':0,
            'MK':1,
            'US':2
        }
        self.All_Point = [
            self.PP_points,
            self.MK_points,
            self.US_points
        ]

    def str_2_index(self,str_x):
        pl = str_x.split('_')
        Name = pl[0]
        index = pl[1]
        ni = self.dict[Name]
        index = int(index)%self.dict_len[Name]
        return ni+index

    def Add_point(self,A_str,B_str):
        ai = self.str_2_index(A_str)
        bi = self.str_2_index(B_str)
        for i in range(3):
            A = self.All_Point[i][ai]
            B = self.All_Point[i][bi]
            C = list((np.array(A)+np.array(B))*0.5)
            self.All_Point[i].append(C)

    def get_point(self):
        return self.All_Point[0],self.All_Point[1],self.All_Point[2]

    def face_ex(self):
        begin = self.dict['F']
        end = begin+self.dict_len['F']

        for i in range(3):
            n = np.array(self.All_Point[i][begin:end])
            a = n[0].reshape(1, 3)
            b = n[-1].reshape(1, 3)
            c = (a+b)/2
            d = (n-c)
            if i==1:
                d[:,1]*=0.7
            else:
                d[:,1]*=1.1
            C = list(c-d)
            self.All_Point[i] =self.All_Point[i] + C

    def revise(self,str_,v):
        su = str_.split('_')
        s = su[0]
        u = su[1]

        begin = self.dict[s]
        end = begin+self.dict_len[s]
        i = self.dict_usr[u]
        n = np.array(self.All_Point[i][begin:end])
        diff= np.array(v).reshape(1,3)
        C = list(n+diff)

        self.All_Point[i][begin:end] = C

    #def polyfill_(self):


if __name__ == '__main__':
    facePoint = 'faceseg.get_face_key_points()'
    fm = FaceModel(facePoint)
    fm.Add_point('LE_0','LE_1')
    fm.Add_point('RE_0', 'RE_1')
    fm.Add_point('N_0', 'M_1')
    fm.revise('LB_MK', [0, -0.18, 0])
    points,test_points,tar_points = fm.get_point()