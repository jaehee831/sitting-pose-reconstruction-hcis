using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Runtime.InteropServices;
using Tao.OpenGl;

namespace MUFLR
{
    public class Sprite
    {
        internal const int VertexAmount = 4;
        Vector[] _pos = new Vector[VertexAmount];
        public Vector CenterLocation = new Vector(0, 0, 0);
        Color[] _col = new Color[VertexAmount];
        Point[] _uv = new Point[VertexAmount];
        AnimSet animSet = null;
        Texture _t = new Texture();
        public Sprite()
        {
            Init(new Vector(0, 0, 0), 1, 1);
            SetColor(new Color(1, 1, 1, 1));
            SetUV(new Point(0, 0), new Point(1, 1));
        }
        public Texture Texture { get { return _t; } set { _t = value; } }
        public Vector[] Positions { get { return _pos; } }
        public Color[] Colors { get { return _col; } }
        public Point[] UVs { get { return _uv; } }
        private Vector GetCenter()
        {
            double halfwidth = GetWidth() / 2;// + Axix.X*GetWidth();
            double halfheight = GetHeight() / 2;// + Axix.Y*GetHeight();
            return new Vector(_pos[1].X + halfwidth, _pos[1].Y + halfheight, _pos[1].Z);
        }
        private void Init(Vector pos, double width, double height)
        {
            double halfwidth = width / 2;
            double halfheight = height / 2;
            CenterLocation = pos;

            _pos[0] = new Vector(pos.X + halfwidth, pos.Y - halfheight, pos.Z);
            _pos[1] = new Vector(pos.X - halfwidth, pos.Y - halfheight, pos.Z);
            _pos[2] = new Vector(pos.X - halfwidth, pos.Y + halfheight, pos.Z);
            _pos[3] = new Vector(pos.X + halfwidth, pos.Y + halfheight, pos.Z);
        }
        public double GetWidth() { return _pos[0].X - _pos[1].X; }
        public double GetHeight() { return _pos[2].Y - _pos[0].Y; }
        public void SetWidth(float width) { Init(GetCenter(), width, GetHeight()); }
        public void SetHeight(float height) { Init(GetCenter(), GetWidth(), height); }
        public void SetPosition(double x, double y) { SetPosition(new Vector(x, y, 0)); }
        public void SetPosition(Vector v) { Init(v, GetWidth(), GetHeight()); }
        public void SetUninterrupt(Vector v, float Width, float Height) { Init(v, Width, Height); }
        public void SetColor(Color color) { for (int i = 0; i < Sprite.VertexAmount; i++) _col[i] = color; }
        public void SetUV(Point topLeft, Point bottomRight)
        {
            _uv[0] = new Point(bottomRight.X, topLeft.Y);
            _uv[1] = topLeft;
            _uv[2] = new Point(topLeft.X, bottomRight.Y);
            _uv[3] = bottomRight;
        }
        public void FlipImage(bool x, bool y)
        {
            if (x)
            {
                Point e = _uv[0];
                _uv[0] = _uv[1];
                _uv[1] = e;
            }
            if (y)
            {
                Point e = _uv[2];
                _uv[2] = _uv[3];
                _uv[3] = e;
            }
        }
        public void RotateAngle(double deg, float Width, float Height)
        {
            Init(CenterLocation, Width, Height);
            deg *= Math.PI; deg /= 180; //Convert to radians
            Vector Center = GetCenter();
            double Theta = Math.Atan2(Height, Width);
            double Hypo = Math.Pow(Math.Pow(Width/2, 2) + Math.Pow(Height/2, 2), .5); //Get hypotenuse
            _pos[0] = new Vector(Center.X + Hypo * Math.Cos(deg - Theta), Center.Y + Hypo * Math.Sin(deg - Theta), Center.Z);
            _pos[1] = new Vector(Center.X + Hypo * Math.Cos(deg + Math.PI + Theta), Center.Y + Hypo * Math.Sin(deg + Math.PI + Theta), Center.Z);
            _pos[2] = new Vector(Center.X + Hypo * Math.Cos(deg + Math.PI - Theta), Center.Y + Hypo * Math.Sin(deg + Math.PI - Theta), Center.Z);
            _pos[3] = new Vector(Center.X + Hypo * Math.Cos(Theta + deg), Center.Y + Hypo * Math.Sin(Theta + deg), Center.Z);
        }
        public void OrbitAngle(double deg, float Width, float Height, Vector Axix)
        {
            Init(CenterLocation, Width, Height);
            deg *= Math.PI; deg /= 180;
            double Theta = Math.Atan2(Axix.Y - _pos[0].Y, Axix.X - _pos[0].X);
            double Hypo = Math.Pow(Math.Pow(Axix.X - _pos[0].X, 2) + Math.Pow(Axix.Y - _pos[0].Y, 2), .5);
            _pos[0] = new Vector(Axix.X + Hypo * Math.Cos(Theta - deg), Axix.Y + Hypo * Math.Sin(Theta - deg), Axix.Z);
            Theta = Math.Atan2(Axix.Y - _pos[1].Y, Axix.X - _pos[1].X);
            Hypo = Math.Pow(Math.Pow(Axix.X - _pos[1].X, 2) + Math.Pow(Axix.Y - _pos[1].Y, 2), .5);
            _pos[1] = new Vector(Axix.X + Hypo * Math.Cos(Theta - deg), Axix.Y + Hypo * Math.Sin(Theta - deg), Axix.Z);
            Theta = Math.Atan2(Axix.Y - _pos[2].Y, Axix.X - _pos[2].X);
            Hypo = Math.Pow(Math.Pow(Axix.X - _pos[2].X, 2) + Math.Pow(Axix.Y - _pos[2].Y, 2), .5);
            _pos[2] = new Vector(Axix.X + Hypo * Math.Cos(Theta - deg), Axix.Y + Hypo * Math.Sin(Theta - deg), Axix.Z);
            Theta = Math.Atan2(Axix.Y - _pos[3].Y, Axix.X - _pos[3].X);
            Hypo = Math.Pow(Math.Pow(Axix.X - _pos[3].X, 2) + Math.Pow(Axix.Y - _pos[3].Y, 2), .5);
            _pos[3] = new Vector(Axix.X + Hypo * Math.Cos(Theta - deg), Axix.Y + Hypo * Math.Sin(Theta - deg), Axix.Z);
        }
        public void RotateRadian(double rad, float Width, float Height)
        {
            Init(CenterLocation, Width, Height);
            Vector Center = GetCenter();
            double Theta = Math.Atan2(Height, Width);
            double Hypo = Math.Pow(Math.Pow(Width/2, 2) + Math.Pow(Height/2, 2), .5); //Get hypotenuse
            _pos[0] = new Vector(Center.X + Hypo * Math.Cos(Theta + rad), Center.Y + Hypo * Math.Sin(Theta + rad), Center.Z);
            _pos[1] = new Vector(Center.X + Hypo * Math.Cos(rad + Math.PI - Theta), Center.Y + Hypo * Math.Sin(rad + Math.PI - Theta), Center.Z);
            _pos[2] = new Vector(Center.X + Hypo * Math.Cos(rad + Math.PI + Theta), Center.Y + Hypo * Math.Sin(rad + Math.PI + Theta), Center.Z);
            _pos[3] = new Vector(Center.X + Hypo * Math.Cos(rad - Theta), Center.Y + Hypo * Math.Sin(rad - Theta), Center.Z);
        }
        public void Toward(Point targ, double dist)
        {
            Vector Center = GetCenter();
            double Theta = Math.Atan2(targ.Y - Center.Y, targ.X - Center.X);
            CenterLocation = new Vector(Center.X + dist * Math.Cos(Theta), Center.Y + dist * Math.Sin(Theta), 0);
            for (int z = 0; z < 4; z++) _pos[z] = new Vector(_pos[z].X + dist * Math.Cos(Theta), _pos[z].Y + dist * Math.Sin(Theta), 0);
        }
        public void Face(Point Target, float Width, float Height)
        {
            Init(CenterLocation, Width, Height);
            Vector Center = GetCenter();
            double deg = Math.Atan2(Target.Y - Center.Y, Target.X - Center.X);
            RotateRadian(deg-Math.PI, Width, Height);
        }
        public double GetDist(Vector Target, Vector Initial)
        {
            double distx = Target.X - Initial.X;
            double disty = Target.Y - Initial.Y;
            return Math.Pow(Math.Pow(distx, 2) + Math.Pow(disty, 2), .5);
        }
        public void Away(Vector targ, double dist)
        {
            Vector Center = GetCenter();
            double Theta = Math.Atan2(targ.Y - Center.Y, targ.X - Center.X);
            CenterLocation = new Vector(Center.X - dist * Math.Cos(Theta), Center.Y - dist * Math.Sin(Theta), 0);
            for (int z = 0; z < 4; z++) _pos[z] = new Vector(_pos[z].X - dist * Math.Cos(Theta), _pos[z].Y - dist * Math.Sin(Theta), 0);
        }
        public void AnimStart(AnimSet Active)
        {
            Active.Counter = 0;
            animSet = Active;
        }
        public void Animate(Random random, double elapsedTime)
        {
            if (animSet != null)
            {
                if (animSet.Counter < animSet.Required) animSet.Counter+=elapsedTime;
                else
                {
                    animSet.Counter = 0;
                    if (animSet.Index <= animSet.Textures.Length)
                    {
                        this._t = animSet.Textures[animSet.Index];
                        this.SetWidth(_t.Width);
                        this.SetHeight(_t.Height);
                    }
                    if (animSet.EndAction != AnimCommand.PlayRandom)
                    {
                        if (animSet.Index == animSet.Textures.Length-1) switch (animSet.EndAction)
                            {
                                case AnimCommand.PlayOnce:
                                    animSet.Index = 0;
                                    animSet = null;
                                    break;
                                case AnimCommand.Repeat:
                                    animSet.Index = 0;
                                    break;
                            }
                        else animSet.Index++;
                    }
                    else animSet.Index = random.Next(0, animSet.Textures.Length);
                }
            }
        }
        public void AnimStop(Texture Set)
        {
            animSet = null;
            this._t = Set;
            this.SetWidth(_t.Width);
            this.SetHeight(_t.Height);
        }
        public AnimSet AnimGet() { return animSet; }
    }
    public enum AnimCommand { PlayOnce, Repeat, PlayRandom }
    public enum Anchor { Left, Right, Up, Down, Center }
    public class AnimSet
    {
        public double Counter = 0;
        public double Required = 1;
        public int Index = 0;
        public Anchor Anchor = Anchor.Center;
        public AnimCommand EndAction = AnimCommand.PlayOnce;
        public Texture[] Textures = {  };
        public AnimSet(double time, Texture[] textures)
        {
            Required = time;
            Counter = time;
            Textures = textures;
        }
        public AnimSet(double time, Texture[] textures, int indexStart)
        {
            Required = time;
            Counter = time;
            Textures = textures;
            Index = indexStart;
        }
        public AnimSet(double time, Texture[] textures, int indexStart, double delay)
        {
            Required = time;
            Counter = time - delay;
            Textures = textures;
            Index = indexStart;
        }
    }
    public class Renderer
    {
        Batch _bath = new Batch();
        public Renderer()
        {
            //Gl.glEnable(Gl.GL_TEXTURE_2D);
            //Gl.glEnable(Gl.GL_BLEND);
            //Gl.glBlendFunc(Gl.GL_SRC_ALPHA, Gl.GL_ONE_MINUS_SRC_ALPHA);
        }
        public void DIMV(Vector pos, Color color, Point uv)
        {
            Gl.glColor4f(color.Red, color.Green, color.Blue, color.Alpha);
            Gl.glTexCoord2f(uv.X, uv.Y);
            Gl.glVertex3d(pos.X, pos.Y, pos.Z);
        }
        int TextureID = -1;
        public void DrawSprite(Sprite S)
        {
            if (S.Texture.ID == TextureID) _bath.AddSprite(S);
            else
            {
                _bath.Draw();

                TextureID = S.Texture.ID;
                Gl.glBindTexture(Gl.GL_TEXTURE_2D, TextureID);
                Gl.glEnable(Gl.GL_BLEND);
                Gl.glBlendFunc(Gl.GL_SRC_ALPHA, Gl.GL_ONE_MINUS_SRC_ALPHA);
                _bath.AddSprite(S);
            }
        }
        public void Render()
        {
            _bath.Draw();
            //Gl.glDisable(Gl.GL_BLEND);
        }
        public void Setup2DGraphics(double width, double height)
        {
            double halfwidth = width / 2;
            double halfheight = height / 2;
            Gl.glMatrixMode(Gl.GL_PROJECTION);
            Gl.glLoadIdentity();
            Gl.glOrtho(-halfwidth, halfwidth, -halfheight, halfheight, -100, 100);
            Gl.glMatrixMode(Gl.GL_MODELVIEW);
            Gl.glLoadIdentity();
        }
    }
    public class Batch
    {
        const int MaxVertexNumber = 1000;
        Vector[] _pos = new Vector[MaxVertexNumber];
        Color[] _col = new Color[MaxVertexNumber];
        Point[] _uv = new Point[MaxVertexNumber];
        int _batchSize = 0;
        public void AddSprite(Sprite sprite)
        {
            if (sprite.Positions.Length + _batchSize > MaxVertexNumber) { Draw(); }
            for (int i = 0; i < sprite.Positions.Length; i++)
            {
                _pos[_batchSize + i] = sprite.Positions[i];
                _col[_batchSize + i] = sprite.Colors[i];
                _uv[_batchSize + i] = sprite.UVs[i];
            }
            //I once had a pet newt. I named it Tiny because it was my newt [minute]
            _batchSize += sprite.Positions.Length;
        }
        const int Dimensions = 3;
        const int ColorDimensions = 4;
        const int UV = 2;
        void SetupPointers()
        {
            Gl.glEnableClientState(Gl.GL_COLOR_ARRAY);
            Gl.glEnableClientState(Gl.GL_VERTEX_ARRAY);
            Gl.glEnableClientState(Gl.GL_TEXTURE_COORD_ARRAY);
            Gl.glVertexPointer(Dimensions, Gl.GL_DOUBLE, 0, _pos);
            Gl.glColorPointer(ColorDimensions, Gl.GL_FLOAT, 0, _col);
            Gl.glTexCoordPointer(UV, Gl.GL_FLOAT, 0, _uv);
        }
        public void Draw()
        {
            if (_batchSize == 0) return;
            SetupPointers();
            Gl.glDrawArrays(Gl.GL_QUADS, 0, _batchSize);
            _batchSize = 0;
        }
    }
    [StructLayout(LayoutKind.Sequential)]
    public struct Vector
    {
        public static Vector Zero = new Vector(0, 0, 0);
        public double X { get; set; }
        public double Y { get; set; }
        public double Z { get; set; }
        public Vector(double x, double y, double z) : this()
        {
            X = x; Y = y; Z = z;
        }
        public override string ToString()
        {
            return string.Format("X:{0}, Y:{1}, Z:{2}", X, Y, Z);
        }
    }
    [StructLayout(LayoutKind.Sequential)]
    public struct Point
    {
        public float X { get; set; }
        public float Y { get; set; }
        public Point(float x, float y) : this()
        {
            X = x; Y = y;
        }
    }
    [StructLayout(LayoutKind.Sequential)]
    public struct Color
    {
        public float Red { get; set; }
        public float Green { get; set; }
        public float Blue { get; set; }
        public float Alpha { get; set; }
        public Color(float r, float g, float b, float a) : this()
        {
            Red = r; Green = g; Blue = b; Alpha = a;
        }
    }
}
