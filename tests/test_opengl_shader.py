import manim.utils.opengl as opengl
from manim import *
from manim.opengl import *  # type: ignore


class InlineShaderExample(Scene):
    def construct(self):
        config["background_color"] = "#333333"

        c = Circle(fill_opacity=0.7).shift(UL)
        self.add(c)

        shader = Shader(
            self.renderer.context,
            source={
                "vertex_shader": """
                #version 330
                in vec4 in_vert;
                in vec4 in_color;
                out vec4 v_color;
                uniform mat4 u_model_view_matrix;
                uniform mat4 u_projection_matrix;
                void main() {
                    v_color = in_color;
                    vec4 camera_space_vertex = u_model_view_matrix * in_vert;
                    vec4 clip_space_vertex = u_projection_matrix * camera_space_vertex;
                    gl_Position = clip_space_vertex;
                }
            """,
                "fragment_shader": """
            #version 330
            in vec4 v_color;
            out vec4 frag_color;
            void main() {
              frag_color = v_color;
            }

            void main () {
                // Previously, you'd have rendered your complete scene into a texture
                // bound to "fullScreenTexture."
                vec4 rValue = texture2D(fullscreenTexture, gl_TexCoords[0] - rOffset);  
                vec4 gValue = texture2D(fullscreenTexture, gl_TexCoords[0] - gOffset);
                vec4 bValue = texture2D(fullscreenTexture, gl_TexCoords[0] - bOffset);  

                // Combine the offset colors.
                gl_FragColor = vec4(rValue.r, gValue.g, bValue.b, 1.0);
            }
            """,
            },
        )
        shader.set_uniform("u_model_view_matrix", opengl.view_matrix())
        shader.set_uniform(
            "u_projection_matrix",
            opengl.orthographic_projection_matrix(),
        )

        attributes = np.zeros(
            6,
            dtype=[
                ("in_vert", np.float32, (4,)),
                ("in_color", np.float32, (4,)),
            ],
        )
        attributes["in_vert"] = np.array(
            [
                [-1, -1, 0, 1],
                [-1, 1, 0, 1],
                [1, 1, 0, 1],
                [-1, -1, 0, 1],
                [1, -1, 0, 1],
                [1, 1, 0, 1],
            ],
        )
        attributes["in_color"] = np.array(
            [
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
            ],
        )
        mesh = Mesh(shader, attributes)
        self.add(mesh)

        self.wait(5)
        # self.embed_2()
