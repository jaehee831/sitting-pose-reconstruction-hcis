using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Windows.Input;
using System.Threading;
using System.IO.Ports;
using System.IO;
using Tao.OpenGl;
using Tao.FreeGlut;
using MUFLR;
using System.Runtime.InteropServices;

namespace TheBird
{
    public partial class Bird : Form
    {
        double[] Data = new double[17];
        bool Sensor = false;
        int COMNum = 1;
        int comState = 0;
        int TimeOut = 500;
        string inBuffer = "";
        Vector moveLock = Vector.Zero;
        Input _i = new Input();
        double x = 0;
        Random r = new Random();
        double y = 0;
        double x0 = 0;
        double y0 = 0;
        double phi = 0;
        double theta = 0;
        double radius = 0;
        double phiDeg = 0;
        double thetaDeg = 0;
        double pitch = 0;
        double roll = 0;
        double[] magCalMin = new double[6];
        double[] magCalMax = new double[6];
        double magRadius = 0;
        double magPhi = 0;
        double magPhiDeg = 0;
        double magTheta = 0;
        double magThetaDeg = 0;
        double magDir = 0;

        bool changed = false;
        public Bird()
        {
            Cycle C = new Cycle(Update);
            InitializeComponent();
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.None;
            GLC.InitializeContexts();
            _i.Mouse = new MUFLR.Mouse(this, GLC);
            Glut.glutInit();
            Glut.glutInitDisplayMode(Glut.GLUT_RGBA | Glut.GLUT_DOUBLE | Glut.GLUT_DEPTH);
            Gl.glMatrixMode(Gl.GL_PROJECTION);
            Gl.glLoadIdentity();
            Gl.glOrtho(-Screen.PrimaryScreen.WorkingArea.Width, Screen.PrimaryScreen.WorkingArea.Width,
                -Screen.PrimaryScreen.WorkingArea.Height, Screen.PrimaryScreen.WorkingArea.Height, -700, 700);
            Gl.glMatrixMode(Gl.GL_MODELVIEW);
            Gl.glLoadIdentity();
            Gl.glEnable(Gl.GL_LIGHTING);
            Gl.glEnable(Gl.GL_LIGHT0);
            Gl.glLightfv(Gl.GL_LIGHT0, Gl.GL_POSITION, new float[3] { -100, -100, 100 });
            Gl.glEnable(Gl.GL_DEPTH_TEST);
            Data = new double[17] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
        }
        public void Update(double elapsedTime)
        {
            if (!Sensor) Search(elapsedTime);
            UpdateInput(elapsedTime);
            float[] e = { 1, 1, 1 };

            //Bird detection
            if ( (Data[0] > 2*32767 / 3) && (Data[2] > 2*32767 / 3) && (Data[4] < 32767 / 2) && (Data[6] > 1*32767 / 2) )
                e = new float[3] { (float)r.Next(0, 10)/10f, (float)r.Next(0, 10)/10f, (float)r.Next(0, 10)/10f };


            sensor0.Text = Data[0].ToString();
            sensor1.Text = Data[1].ToString();

            sensor2.Text = Data[2].ToString();
            sensor3.Text = Data[3].ToString();

            sensor4.Text = Data[4].ToString();
            sensor5.Text = Data[5].ToString();

            sensor6.Text = Data[6].ToString();
            sensor7.Text = Data[7].ToString();

            sensor8.Text = Data[8].ToString();
            sensor9.Text = Data[9].ToString();



            accelX.Text = Data[10].ToString();
            accelY.Text = Data[11].ToString();
            accelZ.Text = Data[12].ToString();

            magX.Text = Data[13].ToString();
            magY.Text = Data[14].ToString();
            magZ.Text = Data[15].ToString();

            
            

            radius = Math.Sqrt(Math.Pow(Data[10], 2) + Math.Pow(Data[11], 2) + Math.Pow(Data[12], 2) );
            theta = Math.Atan(Data[11] / (Data[12] + 1));

            if (Data[11] > 0)
            {
                if (Data[12] > 0)
                {
                    thetaDeg = Math.Abs(theta * 180 / Math.PI);
                }
                else
                {
                    thetaDeg = 180 - Math.Abs(theta * 180 / Math.PI);
                }
            }
            else
            {
                if (Data[12] > 0)
                {
                    thetaDeg = -Math.Abs(theta * 180 / Math.PI);
                }
                else
                {
                    thetaDeg = -180+Math.Abs(theta * 180 / Math.PI);
                }
            }
            
            phi = Math.Acos(Data[10] / radius);

            phiDeg = 90 - phi * 180 / Math.PI;
            

            if (double.IsNaN(phiDeg))
            {
                phiDeg = 0;
                pitch = 0;
            }

            //Smoothing filter
            pitch += (phiDeg - pitch) * 0.02;

            roll += (thetaDeg - roll) * 0.02;

            magRadius = Math.Sqrt(Math.Pow(Data[13], 2) + Math.Pow(Data[14], 2) + Math.Pow(Data[15], 2));
          
            magTheta = Math.Atan(Data[14] / (Data[13]+1) );
            magThetaDeg = magTheta * 180 / Math.PI;

            magPhi = Math.Acos(Data[15] / magRadius);
            magPhiDeg = magPhi * 180 / Math.PI;


            mouseY.Text = magThetaDeg.ToString();
            mouseX.Text = magPhiDeg.ToString();

            Gl.glClearColor(e[0], e[1], e[2], 1);
            Gl.glClear(Gl.GL_COLOR_BUFFER_BIT | Gl.GL_DEPTH_BUFFER_BIT);
            //Gl.glEnable(Gl.GL_ALPHA_TEST);
            Gl.glAlphaFunc(Gl.GL_NOTEQUAL, 0);
            //Gl.glEnable(Gl.GL_BLEND);
            //Gl.glBlendFunc(Gl.GL_SRC_ALPHA, Gl.GL_ONE_MINUS_SRC_ALPHA);
            Gl.glRotated(x, 0, 1, 0);
            Gl.glRotated(pitch, -1, 0, 0);

            //Gl.glRotated(pitch, 1, 0, 0);  //Up down, but opposite direction
            //Gl.glRotated(pitch, 0, 1, 0);    //Rotates about the x-y axis
            
            
            Gl.glRotated(roll, 0, 0, 1);     //roll

            double g = 0;
            //Make Hand
            Gl.glPushMatrix();
            Gl.glColor4d(1, 0, 0, 1);
            Gl.glTranslated(75 + 75, -75 / 2 - 75, 0);
            Glut.glutSolidCube(75);
            Gl.glTranslated(-75, 0, 0);
            Glut.glutSolidCube(75);
            Gl.glTranslated(-75, 0, 0);
            Glut.glutSolidCube(75);
            Gl.glTranslated(-75, 0, 0);
            Glut.glutSolidCube(75);
            Gl.glTranslated(0, 0, 75);
            Glut.glutSolidCube(75);
            Gl.glTranslated(75, 0, 0);
            Glut.glutSolidCube(75);
            Gl.glTranslated(75, 0, 0);
            Glut.glutSolidCube(75);
            Gl.glTranslated(75, 0, 0);
            Glut.glutSolidCube(75);
            Gl.glTranslated(0, 0, 75);
            Glut.glutSolidCube(75);
            Gl.glTranslated(-75, 0, 0);
            Glut.glutSolidCube(75);
            Gl.glTranslated(-75, 0, 0);
            Glut.glutSolidCube(75);
            Gl.glTranslated(-75, 0, 0);
            Glut.glutSolidCube(75);
            g = (double)Data[0] * 90.0 / 32767.0;
            Gl.glRotated(g, 1, 0, 0);
            {
                Gl.glTranslated(0, 0, 37);
                Gl.glRotated(-10, 0, 1, 0);
                Glut.glutSolidCylinder(40, 100, 20, 1);
                Gl.glTranslated(0, 0, 100);
                Gl.glRotated((double)Data[1]*90.0/32767.0, 1, 0, 0);
                Glut.glutSolidCylinder(35, 100, 20, 1);
                Gl.glTranslated(0, 0, 100);
                Gl.glRotated((double)Data[1]*90.0/32767.0, 1, 0, 0);
                Glut.glutSolidCylinder(30, 100, 20, 1);
                Gl.glRotated(-(double)Data[1] * 90.0 / 32767.0, 1, 0, 0);
                Gl.glTranslated(0, 0, -100);
                Gl.glRotated(-(double)Data[1] * 90.0 / 32767.0, 1, 0, 0);
                Gl.glTranslated(0, 0, -100);
                Gl.glRotated(10, 0, 1, 0);
                Gl.glTranslated(0, 37, -37);
                Gl.glRotated(-g, 1, 0, 0);
            }
            Gl.glTranslated(75, -75.0 / 2.0 * (90 - g) / 90.0, 75.0 / 2.0 * (90 - g) / 90.0);
            Gl.glTranslated(0, 0, -75.0 / 2.0);
            g = (double)Data[2] * 90.0 / 32767.0;
            Gl.glRotated(g, 1, 0, 0);
            {
                Gl.glTranslated(0, 0, 75.0 / 2.0);
                Gl.glRotated(-5, 0, 1, 0);
                Glut.glutSolidCylinder(40, 125, 20, 1);
                Gl.glTranslated(0, 0, 125);
                Gl.glRotated((double)Data[3]*90.0/32767.0, 1, 0, 0);
                Glut.glutSolidCylinder(35, 125, 20, 1);
                Gl.glTranslated(0, 0, 125);
                Gl.glRotated((double)Data[3] * 90.0 / 32767.0, 1, 0, 0);
                Glut.glutSolidCylinder(30, 125, 20, 1);
                Gl.glRotated(-(double)Data[3] * 90.0 / 32767.0, 1, 0, 0);
                Gl.glTranslated(0, 0, -125);
                Gl.glRotated(-(double)Data[3] * 90.0 / 32767.0, 1, 0, 0);
                Gl.glTranslated(0, 0, -125);
                Gl.glRotated(5, 0, 1, 0);
                Gl.glRotated(-g, 1, 0, 0);
            }
            Gl.glTranslated(0, -75.0 / 2.0 * (90 - g) / 90, -75.0 / 2.0 * (90 - g) / 90.0);
            Gl.glTranslated(75, 75 / 2.0, 0);
            g = (double)Data[4] * 90.0 / 32767.0;
            Gl.glRotated(g, 1, 0, 0);
            {
                Gl.glTranslated(0, 0, 75 / 2.0);
                Glut.glutSolidCylinder(40, 150, 20, 1);
                Gl.glTranslated(0, 0, 150);
                Gl.glRotated((double)Data[5] * 90.0 / 32767.0, 1, 0, 0);
                Glut.glutSolidCylinder(35, 150, 20, 1);
                Gl.glTranslated(0, 0, 150);
                Gl.glRotated((double)Data[5] * 90.0 / 32767.0, 1, 0, 0);
                Glut.glutSolidCylinder(30, 125, 20, 1);
                Gl.glRotated(-(double)Data[5] * 90.0 / 32767.0, 1, 0, 0);
                Gl.glTranslated(0, 0, -150);
                Gl.glRotated(-(double)Data[5] * 90.0 / 32767.0, 1, 0, 0);
                Gl.glTranslated(0, 0, -150);
                Gl.glRotated(-g, 1, 0, 0);
            }
            Gl.glTranslated(0, -75.0 / 2.0 * (90 - g) / 90, -75.0 / 2.0 * (90 - g) / 90.0);
            Gl.glTranslated(75, 75 / 2.0, 0);
            g = (double)Data[6] * 90.0 / 32767.0;
            Gl.glRotated(g, 1, 0, 0);
            {
                Gl.glTranslated(0, 0, 75 / 2.0);
                Glut.glutSolidCylinder(40, 125, 20, 1);
                Gl.glTranslated(0, 0, 125);
                Gl.glRotated((double)Data[7] * 90.0 / 32767.0, 1, 0, 0);
                Glut.glutSolidCylinder(35, 125, 20, 1);
                Gl.glTranslated(0, 0, 125);
                Gl.glRotated((double)Data[7] * 90.0 / 32767.0, 1, 0, 0);
                Glut.glutSolidCylinder(30, 125, 20, 1);
                Gl.glRotated(-(double)Data[7] * 90.0 / 32767.0, 1, 0, 0);
                Gl.glTranslated(0, 0, -125);
                Gl.glRotated(-(double)Data[7] * 90.0 / 32767.0, 1, 0, 0);
                Gl.glTranslated(0, 0, -125);
                Gl.glRotated(-5, 0, 1, 0);
                Gl.glRotated(-g, 0, 1, 0);
            }

            //Thumb

            Gl.glLoadIdentity();
            Gl.glRotated(x, 0, 1, 0);
            Gl.glRotated(pitch, -1, 0, 0);
            Gl.glRotated(roll, 0, 0, 1);     //roll
            Gl.glTranslated(150, -113, 0);
            Gl.glRotated(90, 0, 1, 0);
            Glut.glutSolidCylinder(37, 50, 20, 1);
            Gl.glTranslated(0, 0, 70);
            Glut.glutSolidSphere(37, 37, 37);
            Gl.glRotated(-40, 0, 1, 0);
            g = (double)Data[8] * 180.0 / 32767.0;
            if (g > 90) g = 90;
            Gl.glRotated(g, 1, 0, 0);
            Glut.glutSolidCylinder(40, 150, 20, 1);
            //Gl.glRotated(-10, 1, 0, 0);
            Gl.glTranslated(0, 0, 150);
            g = (double)Data[8] * 90.0 / 32767.0;
            Gl.glRotated(g, 1, 0, -1);
            Glut.glutSolidCylinder(35, 150, 20, 1);
            Gl.glPopMatrix();
            //</>
            Gl.glLoadIdentity();
            Gl.glFinish();
            GLC.Refresh();

            



        }
        public void Search(double elapsedTime)
        {
            switch (comState)
            {
                case 0:
                    Console.WriteLine(string.Format("Trying port COM{0}...", COMNum));
                    Hand.PortName = "COM" + COMNum.ToString();
                    try
                    {
                        Hand.Open();
                        comState = 1;
                    }
                    catch
                    {
                        COMNum++;
                        if (COMNum > 200)
                        {
                            COMNum = 1;
                            comState = 0;
                        } //Done
                    }
                    break;
                case 1:
                    inBuffer = "";
                    try
                    {
                        Hand.WriteLine("enable");
                        //Hand.WriteLine("enable");
                    }
                    catch { }
            
                        
                    TimeOut = 100;
                    comState = 2;
                    break;
                case 2:
                    if (inBuffer != "")
                    {
                        Sensor = true;
                        comState = 0xFF;
                    }
                    else
                    {
                        if (TimeOut > 0) TimeOut--;
                        else
                        {
                            try
                            {
                                //Hand.WriteLine("enable");
                                //Hand.WriteLine("enable");
                            }
                            catch { }
                        }
                    }
                    break;
            }
        }
        public void UpdateInput(double elapsedTime)
        {
            if (Keyboard.IsKeyDown(Key.Escape)) Application.Exit();
            if (Keyboard.IsKeyDown(Key.A))
            {
                if (Data[0] < 32767) Data[0] += elapsedTime * 10000;
                if (Data[1] < 32767) Data[1] += elapsedTime * 10000;
            }
            if (Keyboard.IsKeyDown(Key.Q))
            {
                if (Data[0] > 0) Data[0] -= elapsedTime * 10000;
                if (Data[1] > 0) Data[1] -= elapsedTime * 10000;
            }
            if (Keyboard.IsKeyDown(Key.S))
            {
                if (Data[3] < 32767) Data[3] += elapsedTime * 10000; 
                if (Data[2] < 32767) Data[2] += elapsedTime * 10000;
            }
            if (Keyboard.IsKeyDown(Key.W))
            {
                if (Data[3] > 0) Data[3] -= elapsedTime * 10000;
                if (Data[2] > 0) Data[2] -= elapsedTime * 10000;
            }
            if (Keyboard.IsKeyDown(Key.D))
            {
                if (Data[5] < 32767) Data[5] += elapsedTime * 10000;
                if (Data[4] < 32767) Data[4] += elapsedTime * 10000;
            }
            if (Keyboard.IsKeyDown(Key.E))
            {
                if (Data[5] > 0) Data[5] -= elapsedTime * 10000;
                if (Data[4] > 0) Data[4] -= elapsedTime * 10000;
            }
            if (Keyboard.IsKeyDown(Key.F))
            {
                if (Data[7] < 32767) Data[7] += elapsedTime * 10000;
                if (Data[6] < 32767) Data[6] += elapsedTime * 10000;
            }
            if (Keyboard.IsKeyDown(Key.R))
            {
                if (Data[7] > 0) Data[7] -= elapsedTime * 10000;
                if (Data[6] > 0) Data[6] -= elapsedTime * 10000;
            }
            if (Keyboard.IsKeyDown(Key.G))
            {
                if (Data[8] < 32767) Data[8] += elapsedTime * 10000;
                if (Data[9] < 32767) Data[9] += elapsedTime * 10000;
            }
            if (Keyboard.IsKeyDown(Key.T))
            {
                if (Data[8] > 0) Data[8] -= elapsedTime * 10000;
                if (Data[9] > 0) Data[9] -= elapsedTime * 10000;
            }
            _i.Update(elapsedTime);
            if (_i.Mouse.LeftHeld || _i.Mouse.RightHeld)
            {
                if (changed)
                {
                    x += (_i.Mouse.Position.X - x0);
                    //if (((x % 360 > 90 && x % 360 < 270) || (y % 360 > 90 && y % 360 < 270))&&
                    //    !((x % 360 > 90 && x % 360 < 270) && (y % 360 > 90 && y % 360 < 270)))
                    //    y -= (_i.Mouse.Position.Y - y0);
                    //else y += (_i.Mouse.Position.Y - y0);
                    //y += (_i.Mouse.Position.Y - y0);
                }
                x0 = _i.Mouse.Position.X;
                y0 = _i.Mouse.Position.Y;
                changed = true;
            }
            else changed = false;
        }
        protected override void OnClientSizeChanged(EventArgs e)
        {
            base.OnClientSizeChanged(e);
            Gl.glViewport(0, 0, this.ClientSize.Width, this.ClientSize.Height);
            Gl.glMatrixMode(Gl.GL_PROJECTION);
            Gl.glLoadIdentity();
            Gl.glOrtho(-Screen.PrimaryScreen.WorkingArea.Width, Screen.PrimaryScreen.WorkingArea.Width,
                -Screen.PrimaryScreen.WorkingArea.Height, Screen.PrimaryScreen.WorkingArea.Height, -700, 700);
            Gl.glMatrixMode(Gl.GL_MODELVIEW);
            Gl.glLoadIdentity();
        }
        private void DataRecieved(object sender, SerialDataReceivedEventArgs e)
        {
            try
            {
                SerialPort sp = (SerialPort)sender;

                if (comState > 0x10)
                {
                    //if (State2 > 10)
                    //{
                    string u = "";
                    string m = sp.ReadLine();
                    Console.WriteLine(m);
                    if (m == "enable") m = sp.ReadLine();
                    Int32[] s = new Int32[30];
                    //Read all integers instead of just one.
                    int h = m.IndexOf(":");
                    int j = m.IndexOf("]");
                    for (int i = 0; i < 17; i++)
                    {
                        for (int z = h + 1; z < j; z++) u += m[z];
                        try { Data[i] = Convert.ToInt32(u); }
                        catch { }
                        string v = "";
                        for (int k = j + 1; k < m.Length; k++) v += m[k];
                        h = v.IndexOf(":");
                        j = v.IndexOf("]");
                        m = v;
                        u = "";
                    }
                    if (!sp.IsOpen) { try { sp.Open(); } catch { } }
                    try { sp.DiscardInBuffer(); }
                    catch { }
                    //}
                    //else
                    //{
                    //    inBuffer = sp.ReadExisting();
                    //}
                }
                else
                {
                    inBuffer = sp.ReadExisting();
                }
            }
            catch { }
        }

        private void Send(object sender, KeyPressEventArgs e)
        {
            if (e.KeyChar == 'o' || e.KeyChar == 'O') if (Sensor)
            {
                try
                {
                    Hand.WriteLine("d");
                    Hand.WriteLine("calmin");
                    Hand.WriteLine("enable");
                }
                catch { }
                
            }
            
            if (e.KeyChar == 'c'||e.KeyChar == 'C') if (Sensor)
            {
                try
                {
                    Hand.WriteLine("d");
                    Hand.WriteLine("calmax");
                    Hand.WriteLine("enable");
                }
                catch { }

            }
        }

        private void label1_Click(object sender, EventArgs e)
        {

        }

        private void label2_Click(object sender, EventArgs e)
        {

        }

        private void label2_Click_1(object sender, EventArgs e)
        {

        }

        private void GLC_Load(object sender, EventArgs e)
        {
            AllocConsole();
        }

        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        static extern bool AllocConsole();

    }
}
