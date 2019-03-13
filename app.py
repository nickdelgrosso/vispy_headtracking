import numpy as np
from vispy import app, gloo, io, geometry
from vispy.util.transforms import translate, rotate, scale, perspective


VERT_SHADER = """
attribute vec3 vertex;
attribute vec3 normal;

uniform mat4 model_matrix;
uniform mat4 normal_matrix;
uniform mat4 view_matrix;
uniform mat4 projection_matrix;

varying vec3 v_normal;
varying vec3 v_position;

void main()
{
    v_normal = normalize(normal_matrix * vec4(normal, 1)).xyz;
    v_position = vec3(model_matrix * vec4(vertex, 1));
    gl_Position = projection_matrix * view_matrix * vec4(v_position, 1);
}
"""


FRAG_SHADER = """
varying vec3 v_normal;
varying vec3 v_position;

void main(){
    vec4 color = vec4(1., 1., 0., 1.);
    vec3 surfaceToLight = normalize(vec3(10.5, 0, 0) - v_position);
    float diffuse_reflection = max(dot(v_normal, surfaceToLight), 0);
    gl_FragColor = clamp(color * diffuse_reflection, 0., 1.);
}
"""

## Create Window (will appear when app.run() is called.)
class Canvas(app.Canvas):

    def __init__(self):
        super().__init__(keys='interactive')
        self.show()

        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)
        self.load_mesh()

        self.rotation = 0.
        app.Timer(1/60., self.on_timer).start()

    def load_mesh(self):
        verts, faces, normals, texcoords = io.read_mesh('mesh.obj')
        data = np.zeros(verts.shape[0], [('vertex', np.float32, 3),
                            ('normal', np.float32, 3)])
        data['vertex'] = verts
        data['normal'] = normals
        vertexbuffer = gloo.VertexBuffer(data)
        self.indices = gloo.IndexBuffer(faces)
        self.program.bind(vertexbuffer)

    def update_scene(self):
        scale_matrix = scale([1, 1, 1])
        rotation_matrix = rotate(self.rotation, [0, 1, 0])
        translation_matrix = translate([0.4, 0, -4])
        model_matrix = scale_matrix @ rotation_matrix @ translation_matrix
        self.program['model_matrix'] = model_matrix
        self.program['normal_matrix'] = np.linalg.inv(model_matrix)
        self.program['view_matrix'] = translate([0, 0, 0])
        self.program['projection_matrix'] = perspective(45, 1.66, 1., 100.)

    def on_draw(self, event):
        gloo.clear([1., 0., 0.])
        gloo.set_state(depth_test=True)
        self.program.draw('triangles', self.indices)

    def on_timer(self, event):
        self.rotation += 1.
        self.update_scene()
        self.update()

## Run App
canvas = Canvas()
app.run()
