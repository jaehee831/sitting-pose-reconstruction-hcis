using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using Tao.OpenGl;
using Tao.Sdl;

namespace MUFLR
{
    public class Input
    {
        public Mouse Mouse { get; set; }
        public List<PAL> PAL = new List<PAL>();
        public int _num = 0;
        public void Update(double elapsedTime)
        {
            Mouse.Update(elapsedTime);
            foreach (PAL p in PAL) p.Update();
        }
        public void InitPAL()
        {
            Sdl.SDL_InitSubSystem(Sdl.SDL_INIT_JOYSTICK);
            int g = Sdl.SDL_NumJoysticks();
            if (g > 0) { _num++; PAL.Add(new PAL(0)); }
            if (g > 1) { _num++; PAL.Add(new PAL(1)); }
        }
        public void ReInit()
        {
            _num = 0;
            int g = Sdl.SDL_NumJoysticks();
            PAL = new List<PAL>();
            if (g > 0) { _num++; PAL.Add(new PAL(0)); }
            if (g > 1) { _num++; PAL.Add(new PAL(1)); }
        }
    }
    public class PAL : IDisposable
    {
        IntPtr _pal;
        public ConStick Left { get; private set; }
        public ConStick Right { get; private set; }
        public ConButton Button2 { get; private set; }
        public ConButton Button3 { get; private set; }
        public ConButton Button1 { get; private set; }
        public ConButton Button4 { get; private set; }

        public ConButton Button5 { get; private set; }
        public ConButton Button6 { get; private set; }

        public ConButton Button9 { get; private set; }
        public ConButton Button10 { get; private set; }
        public PAL(int player)
        {
            _pal = Sdl.SDL_JoystickOpen(player);
            Left = new ConStick(_pal, 0, 1);
            Right = new ConStick(_pal, 2, 3);
            Button2 = new ConButton(_pal, 2);
            Button3 = new ConButton(_pal, 3);
            Button1 = new ConButton(_pal, 1);
            Button4 = new ConButton(_pal, 4);
            Button5 = new ConButton(_pal, 5);
            Button6 = new ConButton(_pal, 6);
            Button9 = new ConButton(_pal, 9);
            Button10 = new ConButton(_pal, 10);
        }
        public void Update()
        {
            Left.Update();
            Right.Update();
            Button2.Update();
            Button3.Update();
            Button1.Update();
            Button4.Update();
            Button5.Update();
            Button6.Update();
            Button9.Update();
            Button10.Update();
        }
        #region IDisposable Members
        public void Dispose()
        {
            Sdl.SDL_JoystickClose(_pal);
        }
        #endregion
    }
    public class ConStick
    {
        IntPtr _pal;
        int _axisX = 0;
        int _axisY = 0;
        float _deadZ = 0.2f;
        public float X { get; private set; }
        public float Y { get; private set; }
        public ConStick(IntPtr pal, int axisX, int axisY)
        {
            _pal = pal;
            _axisX = axisX;
            _axisY = axisY;
        }
        public void Update()
        {
            X = Negto1(Sdl.SDL_JoystickGetAxis(_pal, _axisX));
            Y = Negto1(Sdl.SDL_JoystickGetAxis(_pal, _axisY));
        }
        private float Negto1(short value)
        {
            float output = ((float)value / short.MaxValue);
            output = Math.Min(output, 1.0f);
            output = Math.Max(output, -1.0f);
            if (Math.Abs(output) < _deadZ) output = 0;
            return output;
        }
    }
    public class ConButton
    {
        IntPtr _pal;
        int _buttonId;
        public bool Held { get; private set; }
        public ConButton(IntPtr pal, int id)
        {
            _pal = pal;
            _buttonId = id;
        }
        public void Update()
        {
            byte buttonState = Sdl.SDL_JoystickGetButton(_pal, _buttonId);
            Held = (buttonState == 1);
        }
    }
    public class Mouse
    {
        Form _parentForm;
        Control openGlControl;

        public Point Position { get; set; }

        public bool _leftClickDetect = false, _rightClickDetect = false, _middleClickDetect = false;
        public bool MiddlePressed { get; private set; }
        public bool LeftPressed { get; private set; }
        public bool RightPressed { get; private set; }
        public bool MiddleHeld { get; private set; }
        public bool LeftHeld { get; private set; }
        public bool RightHeld { get; set; }
        public Mouse(Form form, Control GLC)
        {
            _parentForm = form;
            openGlControl = GLC;
            openGlControl.MouseClick += delegate(object obj, MouseEventArgs e)
            {
                if (e.Button == MouseButtons.Left) _leftClickDetect = true;
                else if (e.Button == MouseButtons.Right) _rightClickDetect = true;
                else if (e.Button == MouseButtons.Middle) _middleClickDetect = true;
            };
            openGlControl.MouseDown += delegate(object obj, MouseEventArgs e)
            {
                if (e.Button == MouseButtons.Left) LeftHeld = true;
                else if (e.Button == MouseButtons.Right) RightHeld = true;
                else if (e.Button == MouseButtons.Middle) MiddleHeld = true;
            };
            openGlControl.MouseUp += delegate(object obj, MouseEventArgs e)
            {
                if (e.Button == MouseButtons.Left) LeftHeld = false;
                else if (e.Button == MouseButtons.Right) RightHeld = false;
                else if (e.Button == MouseButtons.Middle) MiddleHeld = false;
            };
            openGlControl.MouseLeave += delegate(object obj, EventArgs e)
            {
                LeftHeld = false;
                RightHeld = false;
                MiddleHeld = false;
            };
        }
        public void Update(double elapsedTime)
        {
            UpdateMousePosition();
            UpdateMouseButtons();
        }
        private void UpdateMousePosition()
        {
            System.Drawing.Point MousePos = Cursor.Position;
            MousePos = openGlControl.PointToClient(MousePos);
            MUFLR.Point adjust = new MUFLR.Point();
            adjust.X = (float)MousePos.X - ((float)_parentForm.ClientSize.Width / 2);
            adjust.Y = ((float)_parentForm.ClientSize.Height / 2) - (float)MousePos.Y;
            Position = adjust;
        }
        private void UpdateMouseButtons()
        {
            MiddlePressed = false; LeftPressed = false; RightPressed = false;
            if (_leftClickDetect)
            {
                LeftPressed = true;
                _leftClickDetect = false;
            }
            if (_rightClickDetect)
            {
                RightPressed = true;
                _rightClickDetect = false;
            }
            if (_middleClickDetect)
            {
                MiddlePressed = true;
                _middleClickDetect = false;
            }
        }
    }
}
