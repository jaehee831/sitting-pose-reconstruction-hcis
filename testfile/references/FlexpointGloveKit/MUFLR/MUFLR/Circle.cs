using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tao.OpenGl;

namespace MUFLR
{
    public class Circle
    {
        public Vector Position { get; set; }
        public double Radius { get; set; }
        public float Width { get; set; }
        public Color Color { get; set; }
        public Circle()
        {
            Position = Vector.Zero;
            Width = 1;
            Color = new Color(1, 1, 1, 1);
            Radius = 1;
        }
        public Circle(Vector position, double radius, float width, Color c)
        {
            Position = position;
            Width = width;
            Color = c;
            Radius = radius;
        }
        public Circle(MUFLR.Point position, double radius, float width, Color c)
        {
            Position = new Vector(position.X, position.Y, 0);
            Width = width;
            Color = c;
            Radius = radius;
        }
        public void Draw()
        {
            int vertexAmount = 50;
            double twoPI = 2.0 * Math.PI;
            Gl.glLineWidth(Width);
            Gl.glEnable(Gl.GL_LINE_SMOOTH);
            Gl.glColor4d(Color.Red, Color.Green, Color.Blue, Color.Alpha);
            Gl.glBegin(Gl.GL_LINE_LOOP);
            {
                for (int i = 0; i <= vertexAmount; i++)
                {
                    double xPos = Position.X + Radius * Math.Cos((i * twoPI / vertexAmount));
                    double yPos = Position.Y + Radius * Math.Sin((i * twoPI / vertexAmount));
                    Gl.glVertex2d(xPos, yPos);
                }
            }
            Gl.glEnd();
        }
    }
    public class Arc
    {
        public Vector Position { get; set; }
        public double Radius { get; set; }
        public double DegreeS { get; set; }
        public double DegreeF { get; set; }
        public float Width { get; set; }
        public Color Color { get; set; }
        public int Vertex = 160;
        public Arc()
        {
            Position = Vector.Zero;
            Width = 1;
            Color = new Color(1, 1, 1, 1);
            Radius = 1;
            DegreeS = 0;
            DegreeF = Vertex;
        }
        public Arc(Vector position, double radius, float width, Color c, int Start, int Finish)
        {
            Position = position;
            Width = width;
            Color = c;
            Radius = radius;
            DegreeS = Start;
            DegreeF = Finish;
            while (DegreeS > Vertex) DegreeF -= Vertex;
            while (DegreeS < 0) DegreeF += Vertex;
            while (DegreeF > Vertex) DegreeF -= Vertex;
            while (DegreeF < 0) DegreeF += Vertex;
        }
        public Arc(MUFLR.Point position, double radius, float width, Color c, int Start, int Finish)
        {
            Position = new Vector(position.X, position.Y, 0);
            Width = width;
            Color = c;
            Radius = radius;
            DegreeS = Start;
            DegreeF = Finish;
            while (DegreeS > Vertex) DegreeF -= Vertex;
            while (DegreeS < 0) DegreeF += Vertex;
            while (DegreeF > Vertex) DegreeF -= Vertex;
            while (DegreeF < 0) DegreeF += Vertex;
        }
        public void Draw()
        {
            int vertexAmount = Vertex;
            double twoPI = 2.0 * Math.PI;
            Gl.glLineWidth(Width);
            Gl.glEnable(Gl.GL_LINE_SMOOTH);
            Gl.glEnable(Gl.GL_BLEND);
            Gl.glBlendFunc(Gl.GL_SRC_ALPHA, Gl.GL_ONE_MINUS_SRC_ALPHA);
            Gl.glColor4d(Color.Red, Color.Green, Color.Blue, Color.Alpha);
            Gl.glBegin(Gl.GL_LINE_STRIP);
            {
                for (double i = DegreeS; i <= DegreeF; i++)
                {
                    double xPos = Position.X + Radius * Math.Cos((i * twoPI / vertexAmount));
                    double yPos = Position.Y + Radius * Math.Sin((i * twoPI / vertexAmount));
                    Gl.glVertex2d(xPos, yPos);
                }
            }
            Gl.glEnd();
        }
    }
    public class CircleC
    {
        public Vector Position { get; set; }
        public double O { get; set; }
        public double I { get; set; }
        public Color OuterC { get; set; }
        public Vector[,] Triangles = new Vector[50, 6];
        public Color InnerC { get; set; }
        public double Begin = 0;
        public double End = 50;
        public int Vertix = 50;
        public double Addon = 0;
        public CircleC(double outer, double inner, Vector pos, Color incol, Color outcol, double begin, double end, int vertix, double add)
        {
            Vertix = vertix;
            Triangles = new Vector[Vertix, 6];
            Position = pos;
            O = outer;
            I = inner;
            OuterC = outcol;
            InnerC = incol;
            Begin = begin;
            End = end;
            Addon = add;
            while (Begin > Vertix) End -= Vertix;
            while (Begin < 0) End += Vertix;
            while (End > Vertix) End -= Vertix;
            while (End < 0) End += Vertix;
            double twoPI = 2.0 * Math.PI;
            for (int z = (int)Begin; z < (int)End; z++)
            {
                Triangles[z, 0] = new Vector(outer * Math.Cos((double)z * twoPI / (double)Vertix+Addon) + pos.X, outer * Math.Sin((double)z * twoPI / (double)Vertix+Addon) + pos.Y, 0);
                Triangles[z, 1] = new Vector(inner * Math.Cos((double)z * twoPI / (double)Vertix+Addon) + pos.X, inner * Math.Sin((double)z * twoPI / (double)Vertix+Addon) + pos.Y, 0);
                Triangles[z, 2] = new Vector(outer * Math.Cos((double)(z + 1) * twoPI / (double)Vertix+Addon) + pos.X, outer * Math.Sin((double)(z + 1) * twoPI / (double)Vertix+Addon) + pos.Y, 0);
                Triangles[z, 3] = new Vector(inner * Math.Cos((double)(z + 1) * twoPI / (double)Vertix+Addon) + pos.X, inner * Math.Sin((double)(z + 1) * twoPI / (double)Vertix+Addon) + pos.Y, 0);
                Triangles[z, 4] = new Vector(inner * Math.Cos((double)z * twoPI / (double)Vertix+Addon) + pos.X, inner * Math.Sin((double)z * twoPI / (double)Vertix+Addon) + pos.Y, 0);
                Triangles[z, 5] = new Vector(outer * Math.Cos((double)(z + 1) * twoPI / (double)Vertix+Addon) + pos.X, outer * Math.Sin((double)(z + 1) * twoPI / (double)Vertix+Addon) + pos.Y, 0);
            }
        }
        public void Draw()
        {
            int vertexAmount = Vertix;
            Gl.glBegin(Gl.GL_TRIANGLES);
            Gl.glColor4f(OuterC.Red, OuterC.Green, OuterC.Blue, OuterC.Alpha);
            for (int z = (int)Begin; z < (int)End; z++)
            {
                Gl.glVertex2d(Triangles[z, 0].X, Triangles[z, 0].Y);
                Gl.glColor4f(InnerC.Red, InnerC.Green, InnerC.Blue, InnerC.Alpha);
                Gl.glVertex2d(Triangles[z, 1].X, Triangles[z, 1].Y);
                Gl.glColor4f(OuterC.Red, OuterC.Green, OuterC.Blue, OuterC.Alpha);
                Gl.glVertex2d(Triangles[z, 2].X, Triangles[z, 2].Y);
                Gl.glColor4f(InnerC.Red, InnerC.Green, InnerC.Blue, InnerC.Alpha);
                Gl.glVertex2d(Triangles[z, 3].X, Triangles[z, 3].Y);
                Gl.glVertex2d(Triangles[z, 4].X, Triangles[z, 4].Y);
                Gl.glColor4f(OuterC.Red, OuterC.Green, OuterC.Blue, OuterC.Alpha);
                Gl.glVertex2d(Triangles[z, 5].X, Triangles[z, 5].Y);
            }
            Gl.glEnd();
        }
        public void Rotate(double Radians)
        {
            double twoPI = 2.0 * Math.PI;
            Addon += Radians;
            for (int z = (int)Begin; z < (int)End; z++)
            {
                Triangles[z, 0] = new Vector(O * Math.Cos((double)z * twoPI / (double)Vertix + Addon) + Position.X, O * Math.Sin((double)z * twoPI / (double)Vertix + Addon) + Position.Y, 0);
                Triangles[z, 1] = new Vector(I * Math.Cos((double)z * twoPI / (double)Vertix + Addon) + Position.X, I * Math.Sin((double)z * twoPI / (double)Vertix + Addon) + Position.Y, 0);
                Triangles[z, 2] = new Vector(O * Math.Cos((double)(z + 1) * twoPI / (double)Vertix + Addon) + Position.X, O * Math.Sin((double)(z + 1) * twoPI / (double)Vertix + Addon) + Position.Y, 0);
                Triangles[z, 3] = new Vector(I * Math.Cos((double)(z + 1) * twoPI / (double)Vertix + Addon) + Position.X, I * Math.Sin((double)(z + 1) * twoPI / (double)Vertix + Addon) + Position.Y, 0);
                Triangles[z, 4] = new Vector(I * Math.Cos((double)z * twoPI / (double)Vertix + Addon) + Position.X, I * Math.Sin((double)z * twoPI / (double)Vertix + Addon) + Position.Y, 0);
                Triangles[z, 5] = new Vector(O * Math.Cos((double)(z + 1) * twoPI / (double)Vertix + Addon) + Position.X, O * Math.Sin((double)(z + 1) * twoPI / (double)Vertix + Addon) + Position.Y, 0);
            }
        }
    }
    public class Scalene
    {
        public Vector Position { get; set; }
        public double[] O { get; set; }
        public double[] I { get; set; }
        public Color OuterC { get; set; }
        public Vector[,] Triangles = new Vector[50, 6];
        public Color InnerC { get; set; }
        public double Begin = 0;
        public double End = 50;
        public int Vertix = 50;
        public double Addon = 0;
        public Scalene(double[] outer, double[] inner, Vector pos, Color incol, Color outcol, double begin, double end, int vertix, double add)
        {
            Vertix = vertix;
            Triangles = new Vector[Vertix, 6];
            Position = pos;
            O = outer;
            I = inner;
            OuterC = outcol;
            InnerC = incol;
            Begin = begin;
            End = end;
            Addon = add;
            while (Begin > Vertix) End -= Vertix;
            while (Begin < 0) End += Vertix;
            while (End > Vertix) End -= Vertix;
            while (End < 0) End += Vertix;
            double twoPI = 2.0 * Math.PI;
            for (int z = (int)Begin; z < (int)End; z++)
            {
                Triangles[z, 0] = new Vector(outer[z] * Math.Cos((double)z * twoPI / (double)Vertix + Addon) + pos.X, outer[z] * Math.Sin((double)z * twoPI / (double)Vertix + Addon) + pos.Y, 0);
                Triangles[z, 1] = new Vector(inner[z] * Math.Cos((double)z * twoPI / (double)Vertix + Addon) + pos.X, inner[z] * Math.Sin((double)z * twoPI / (double)Vertix + Addon) + pos.Y, 0);
                Triangles[z, 2] = new Vector(outer[z] * Math.Cos((double)(z + 1) * twoPI / (double)Vertix + Addon) + pos.X, outer[z] * Math.Sin((double)(z + 1) * twoPI / (double)Vertix + Addon) + pos.Y, 0);
                Triangles[z, 3] = new Vector(inner[z] * Math.Cos((double)(z + 1) * twoPI / (double)Vertix + Addon) + pos.X, inner[z] * Math.Sin((double)(z + 1) * twoPI / (double)Vertix + Addon) + pos.Y, 0);
                Triangles[z, 4] = new Vector(inner[z] * Math.Cos((double)z * twoPI / (double)Vertix + Addon) + pos.X, inner[z] * Math.Sin((double)z * twoPI / (double)Vertix + Addon) + pos.Y, 0);
                Triangles[z, 5] = new Vector(outer[z] * Math.Cos((double)(z + 1) * twoPI / (double)Vertix + Addon) + pos.X, outer[z] * Math.Sin((double)(z + 1) * twoPI / (double)Vertix + Addon) + pos.Y, 0);
            }
        }
        public void Draw()
        {
            int vertexAmount = Vertix;
            Gl.glBegin(Gl.GL_TRIANGLES);
            Gl.glColor4f(OuterC.Red, OuterC.Green, OuterC.Blue, OuterC.Alpha);
            for (int z = (int)Begin; z < (int)End; z++)
            {
                Gl.glVertex2d(Triangles[z, 0].X, Triangles[z, 0].Y);
                Gl.glColor4f(InnerC.Red, InnerC.Green, InnerC.Blue, InnerC.Alpha);
                Gl.glVertex2d(Triangles[z, 1].X, Triangles[z, 1].Y);
                Gl.glColor4f(OuterC.Red, OuterC.Green, OuterC.Blue, OuterC.Alpha);
                Gl.glVertex2d(Triangles[z, 2].X, Triangles[z, 2].Y);
                Gl.glColor4f(InnerC.Red, InnerC.Green, InnerC.Blue, InnerC.Alpha);
                Gl.glVertex2d(Triangles[z, 3].X, Triangles[z, 3].Y);
                Gl.glVertex2d(Triangles[z, 4].X, Triangles[z, 4].Y);
                Gl.glColor4f(OuterC.Red, OuterC.Green, OuterC.Blue, OuterC.Alpha);
                Gl.glVertex2d(Triangles[z, 5].X, Triangles[z, 5].Y);
            }
            Gl.glEnd();
        }
        public void Rotate(double Radians)
        {
            double twoPI = 2.0 * Math.PI;
            Addon += Radians;
            for (int z = (int)Begin; z < (int)End; z++)
            {
                Triangles[z, 0] = new Vector(O[z] * Math.Cos((double)z * twoPI / (double)Vertix + Addon) + Position.X, O[z] * Math.Sin((double)z * twoPI / (double)Vertix + Addon) + Position.Y, 0);
                Triangles[z, 1] = new Vector(I[z] * Math.Cos((double)z * twoPI / (double)Vertix + Addon) + Position.X, I[z] * Math.Sin((double)z * twoPI / (double)Vertix + Addon) + Position.Y, 0);
                Triangles[z, 2] = new Vector(O[z] * Math.Cos((double)(z + 1) * twoPI / (double)Vertix + Addon) + Position.X, O[z] * Math.Sin((double)(z + 1) * twoPI / (double)Vertix + Addon) + Position.Y, 0);
                Triangles[z, 3] = new Vector(I[z] * Math.Cos((double)(z + 1) * twoPI / (double)Vertix + Addon) + Position.X, I[z] * Math.Sin((double)(z + 1) * twoPI / (double)Vertix + Addon) + Position.Y, 0);
                Triangles[z, 4] = new Vector(I[z] * Math.Cos((double)z * twoPI / (double)Vertix + Addon) + Position.X, I[z] * Math.Sin((double)z * twoPI / (double)Vertix + Addon) + Position.Y, 0);
                Triangles[z, 5] = new Vector(O[z] * Math.Cos((double)(z + 1) * twoPI / (double)Vertix + Addon) + Position.X, O[z] * Math.Sin((double)(z + 1) * twoPI / (double)Vertix + Addon) + Position.Y, 0);
            }
        }
    }
    public class Polygon
    {
        public List<MUFLR.Point> Positions = new List<MUFLR.Point>();
        public int Width { get; set; }
        public Color Color { get; set; }
        public MUFLR.Point Center { get; set; }
        public double Angle = 0;
        public Polygon(List<MUFLR.Point> Points, int LineWidth, Color color)
        {
            Color = color;
            Width = LineWidth;
            Positions = Points;
            float MedianX = 0;
            foreach (MUFLR.Point f in Positions) MedianX += f.X;
            MedianX /= Positions.Count;
            float MedianY = 0;
            foreach (MUFLR.Point f in Positions) MedianY += f.Y;
            MedianY /= Positions.Count;
            Center = new MUFLR.Point(MedianX, MedianY);
        }
        public void Move(MUFLR.Point e)
        {
            float X = e.X - Center.X;
            float Y = e.Y - Center.Y;
            for (int z = 0; z < Positions.Count; z++) Positions[z] = new MUFLR.Point(Positions[z].X + X, Positions[z].Y + Y);
            Center = new MUFLR.Point(Center.X + X, Center.Y + Y);
        }
        public void RotateAngle(double deg, float Width, float Height)
        {
            deg *= Math.PI; deg /= 180; //Convert to radians
            deg += Angle;
            double Theta = Math.Atan2(Height, Width);
            double Hypo = Math.Pow(Math.Pow(Width / 2, 2) + Math.Pow(Height / 2, 2), .5); //Get hypotenuse
            Positions[0] = new MUFLR.Point((float)(Center.X + Hypo * Math.Cos(deg - Theta)), (float)(Center.Y + Hypo * Math.Sin(deg - Theta)));
            Positions[1] = new MUFLR.Point((float)(Center.X + Hypo * Math.Cos(deg + Math.PI + Theta)), (float)(Center.Y + Hypo * Math.Sin(deg + Math.PI + Theta)));
            Positions[2] = new MUFLR.Point((float)(Center.X + Hypo * Math.Cos(deg + Math.PI - Theta)), (float)(Center.Y + Hypo * Math.Sin(deg + Math.PI - Theta)));
            Positions[3] = new MUFLR.Point((float)(Center.X + Hypo * Math.Cos(Theta + deg)), (float)(Center.Y + Hypo * Math.Sin(Theta + deg)));
        }
        public void Draw()
        {
            Gl.glLineWidth(Width);
            Gl.glEnable(Gl.GL_LINE_SMOOTH);
            Gl.glEnable(Gl.GL_BLEND);
            Gl.glBlendFunc(Gl.GL_SRC_ALPHA, Gl.GL_ONE_MINUS_SRC_ALPHA);
            Gl.glColor4d(Color.Red, Color.Green, Color.Blue, Color.Alpha);
            Gl.glBegin(Gl.GL_LINE_LOOP);
            {
                foreach (MUFLR.Point f in this.Positions) Gl.glVertex2d(f.X, f.Y);
            }
            Gl.glEnd();
        }
    }
}
