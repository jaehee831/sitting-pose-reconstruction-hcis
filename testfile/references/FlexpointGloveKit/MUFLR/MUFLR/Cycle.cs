using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Runtime.InteropServices;
using System.Windows.Forms;

namespace MUFLR
{
    public class Tocker
    {
        [System.Security.SuppressUnmanagedCodeSecurity]
        [DllImport("kernel32")]
        private static extern bool QueryPerformanceFrequency(ref long PerformanceFrequency);
        [System.Security.SuppressUnmanagedCodeSecurity]
        [DllImport("kernel32")]
        private static extern bool QueryPerformanceCounter(ref long PerformanceCount);
        long _ticksPerSecond = 0;
        long _elapsed = 0;
        public Tocker()
        {
            QueryPerformanceFrequency(ref _ticksPerSecond);
            GetElapsedTime();

        }
        public double GetElapsedTime()
        {
            long time = 0;
            QueryPerformanceCounter(ref time);
            double ElapsedTime = (double)(time - _elapsed) / (double)_ticksPerSecond;
            _elapsed = time;
            return ElapsedTime;
        }
    }
    [StructLayout(LayoutKind.Sequential)]
    public struct Message
    {
        public IntPtr hWnd;
        public Int32 msg;
        public IntPtr wParam;
        public IntPtr lParam;
        public uint time;
        public System.Drawing.Point p;
    }
    public class Cycle
    {
        [System.Security.SuppressUnmanagedCodeSecurity]
        [DllImport("User32.dll", CharSet = CharSet.Auto)]
        public static extern bool PeekMessage
        (
            out Message msg,
            IntPtr hWnd,
            uint messageFilterMin,
            uint messageFilterMax,
            uint flags
        );
        Tocker tock = new Tocker();
        public delegate void Callback(double elapsedTime);
        Callback _c;
        public Cycle(Callback c)
        {
            _c = c;
            Application.Idle += new EventHandler(IdleEvent);
        }
        void IdleEvent(object sender, EventArgs e)
        {
            while (StillIdle())
            {
                _c(tock.GetElapsedTime());
            }
        }
        private bool StillIdle()
        {
            Message msg;
            return !PeekMessage(out msg, IntPtr.Zero, 0, 0, 0);
        }
    }
}
