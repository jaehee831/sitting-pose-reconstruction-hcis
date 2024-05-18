namespace TheBird
{
    partial class Bird
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.components = new System.ComponentModel.Container();
            this.GLC = new Tao.Platform.Windows.SimpleOpenGlControl();
            this.Hand = new System.IO.Ports.SerialPort(this.components);
            this.accelX = new System.Windows.Forms.Label();
            this.accelY = new System.Windows.Forms.Label();
            this.magX = new System.Windows.Forms.Label();
            this.accelZ = new System.Windows.Forms.Label();
            this.magZ = new System.Windows.Forms.Label();
            this.magY = new System.Windows.Forms.Label();
            this.mouseY = new System.Windows.Forms.Label();
            this.mouseX = new System.Windows.Forms.Label();
            this.sensor0 = new System.Windows.Forms.Label();
            this.sensor1 = new System.Windows.Forms.Label();
            this.sensor3 = new System.Windows.Forms.Label();
            this.sensor2 = new System.Windows.Forms.Label();
            this.sensor5 = new System.Windows.Forms.Label();
            this.sensor4 = new System.Windows.Forms.Label();
            this.sensor7 = new System.Windows.Forms.Label();
            this.sensor6 = new System.Windows.Forms.Label();
            this.sensor9 = new System.Windows.Forms.Label();
            this.sensor8 = new System.Windows.Forms.Label();
            this.SuspendLayout();
            // 
            // GLC
            // 
            this.GLC.AccumBits = ((byte)(0));
            this.GLC.AutoCheckErrors = false;
            this.GLC.AutoFinish = false;
            this.GLC.AutoMakeCurrent = true;
            this.GLC.AutoSwapBuffers = true;
            this.GLC.BackColor = System.Drawing.Color.White;
            this.GLC.ColorBits = ((byte)(32));
            this.GLC.DepthBits = ((byte)(16));
            this.GLC.Dock = System.Windows.Forms.DockStyle.Fill;
            this.GLC.Location = new System.Drawing.Point(0, 0);
            this.GLC.Name = "GLC";
            this.GLC.Size = new System.Drawing.Size(284, 261);
            this.GLC.StencilBits = ((byte)(0));
            this.GLC.TabIndex = 0;
            this.GLC.Load += new System.EventHandler(this.GLC_Load);
            this.GLC.KeyPress += new System.Windows.Forms.KeyPressEventHandler(this.Send);
            // 
            // Hand
            // 
            this.Hand.WriteTimeout = 100;
            this.Hand.DataReceived += new System.IO.Ports.SerialDataReceivedEventHandler(this.DataRecieved);
            // 
            // accelX
            // 
            this.accelX.AutoSize = true;
            this.accelX.Location = new System.Drawing.Point(12, 9);
            this.accelX.Name = "accelX";
            this.accelX.Size = new System.Drawing.Size(40, 13);
            this.accelX.TabIndex = 1;
            this.accelX.Text = "accelX";
            this.accelX.Visible = false;
            // 
            // accelY
            // 
            this.accelY.AutoSize = true;
            this.accelY.Location = new System.Drawing.Point(12, 34);
            this.accelY.Name = "accelY";
            this.accelY.Size = new System.Drawing.Size(40, 13);
            this.accelY.TabIndex = 2;
            this.accelY.Text = "accelY";
            this.accelY.Visible = false;
            // 
            // magX
            // 
            this.magX.AutoSize = true;
            this.magX.Location = new System.Drawing.Point(12, 82);
            this.magX.Name = "magX";
            this.magX.Size = new System.Drawing.Size(34, 13);
            this.magX.TabIndex = 4;
            this.magX.Text = "magX";
            this.magX.Visible = false;
            // 
            // accelZ
            // 
            this.accelZ.AutoSize = true;
            this.accelZ.Location = new System.Drawing.Point(12, 57);
            this.accelZ.Name = "accelZ";
            this.accelZ.Size = new System.Drawing.Size(40, 13);
            this.accelZ.TabIndex = 3;
            this.accelZ.Text = "accelZ";
            this.accelZ.Visible = false;
            // 
            // magZ
            // 
            this.magZ.AutoSize = true;
            this.magZ.Location = new System.Drawing.Point(12, 130);
            this.magZ.Name = "magZ";
            this.magZ.Size = new System.Drawing.Size(34, 13);
            this.magZ.TabIndex = 6;
            this.magZ.Text = "magZ";
            this.magZ.Visible = false;
            // 
            // magY
            // 
            this.magY.AutoSize = true;
            this.magY.Location = new System.Drawing.Point(12, 105);
            this.magY.Name = "magY";
            this.magY.Size = new System.Drawing.Size(34, 13);
            this.magY.TabIndex = 5;
            this.magY.Text = "magY";
            this.magY.Visible = false;
            // 
            // mouseY
            // 
            this.mouseY.AutoSize = true;
            this.mouseY.Location = new System.Drawing.Point(12, 189);
            this.mouseY.Name = "mouseY";
            this.mouseY.Size = new System.Drawing.Size(45, 13);
            this.mouseY.TabIndex = 8;
            this.mouseY.Text = "mouseY";
            this.mouseY.Visible = false;
            // 
            // mouseX
            // 
            this.mouseX.AutoSize = true;
            this.mouseX.Location = new System.Drawing.Point(12, 164);
            this.mouseX.Name = "mouseX";
            this.mouseX.Size = new System.Drawing.Size(45, 13);
            this.mouseX.TabIndex = 7;
            this.mouseX.Text = "mouseX";
            this.mouseX.Visible = false;
            // 
            // sensor0
            // 
            this.sensor0.AutoSize = true;
            this.sensor0.Location = new System.Drawing.Point(97, 9);
            this.sensor0.Name = "sensor0";
            this.sensor0.Size = new System.Drawing.Size(44, 13);
            this.sensor0.TabIndex = 9;
            this.sensor0.Text = "sensor0";
            this.sensor0.Visible = false;
            this.sensor0.Click += new System.EventHandler(this.label1_Click);
            // 
            // sensor1
            // 
            this.sensor1.AutoSize = true;
            this.sensor1.Location = new System.Drawing.Point(97, 34);
            this.sensor1.Name = "sensor1";
            this.sensor1.Size = new System.Drawing.Size(44, 13);
            this.sensor1.TabIndex = 10;
            this.sensor1.Text = "sensor1";
            this.sensor1.Visible = false;
            // 
            // sensor3
            // 
            this.sensor3.AutoSize = true;
            this.sensor3.Location = new System.Drawing.Point(97, 82);
            this.sensor3.Name = "sensor3";
            this.sensor3.Size = new System.Drawing.Size(44, 13);
            this.sensor3.TabIndex = 12;
            this.sensor3.Text = "sensor3";
            this.sensor3.Visible = false;
            // 
            // sensor2
            // 
            this.sensor2.AutoSize = true;
            this.sensor2.Location = new System.Drawing.Point(97, 57);
            this.sensor2.Name = "sensor2";
            this.sensor2.Size = new System.Drawing.Size(44, 13);
            this.sensor2.TabIndex = 11;
            this.sensor2.Text = "sensor2";
            this.sensor2.Visible = false;
            this.sensor2.Click += new System.EventHandler(this.label2_Click);
            // 
            // sensor5
            // 
            this.sensor5.AutoSize = true;
            this.sensor5.Location = new System.Drawing.Point(97, 130);
            this.sensor5.Name = "sensor5";
            this.sensor5.Size = new System.Drawing.Size(44, 13);
            this.sensor5.TabIndex = 14;
            this.sensor5.Text = "sensor5";
            this.sensor5.Visible = false;
            // 
            // sensor4
            // 
            this.sensor4.AutoSize = true;
            this.sensor4.Location = new System.Drawing.Point(97, 105);
            this.sensor4.Name = "sensor4";
            this.sensor4.Size = new System.Drawing.Size(44, 13);
            this.sensor4.TabIndex = 13;
            this.sensor4.Text = "sensor4";
            this.sensor4.Visible = false;
            // 
            // sensor7
            // 
            this.sensor7.AutoSize = true;
            this.sensor7.Location = new System.Drawing.Point(97, 180);
            this.sensor7.Name = "sensor7";
            this.sensor7.Size = new System.Drawing.Size(44, 13);
            this.sensor7.TabIndex = 16;
            this.sensor7.Text = "sensor7";
            this.sensor7.Visible = false;
            // 
            // sensor6
            // 
            this.sensor6.AutoSize = true;
            this.sensor6.Location = new System.Drawing.Point(97, 155);
            this.sensor6.Name = "sensor6";
            this.sensor6.Size = new System.Drawing.Size(44, 13);
            this.sensor6.TabIndex = 15;
            this.sensor6.Text = "sensor6";
            this.sensor6.Visible = false;
            this.sensor6.Click += new System.EventHandler(this.label2_Click_1);
            // 
            // sensor9
            // 
            this.sensor9.AutoSize = true;
            this.sensor9.Location = new System.Drawing.Point(97, 233);
            this.sensor9.Name = "sensor9";
            this.sensor9.Size = new System.Drawing.Size(44, 13);
            this.sensor9.TabIndex = 18;
            this.sensor9.Text = "sensor9";
            this.sensor9.Visible = false;
            // 
            // sensor8
            // 
            this.sensor8.AutoSize = true;
            this.sensor8.Location = new System.Drawing.Point(97, 208);
            this.sensor8.Name = "sensor8";
            this.sensor8.Size = new System.Drawing.Size(44, 13);
            this.sensor8.TabIndex = 17;
            this.sensor8.Text = "sensor8";
            this.sensor8.Visible = false;
            // 
            // Bird
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(284, 261);
            this.Controls.Add(this.sensor9);
            this.Controls.Add(this.sensor8);
            this.Controls.Add(this.sensor7);
            this.Controls.Add(this.sensor6);
            this.Controls.Add(this.sensor5);
            this.Controls.Add(this.sensor4);
            this.Controls.Add(this.sensor3);
            this.Controls.Add(this.sensor2);
            this.Controls.Add(this.sensor1);
            this.Controls.Add(this.sensor0);
            this.Controls.Add(this.mouseY);
            this.Controls.Add(this.mouseX);
            this.Controls.Add(this.magZ);
            this.Controls.Add(this.magY);
            this.Controls.Add(this.magX);
            this.Controls.Add(this.accelZ);
            this.Controls.Add(this.accelY);
            this.Controls.Add(this.accelX);
            this.Controls.Add(this.GLC);
            this.Name = "Bird";
            this.Text = "FlexpointGloveKit 1.0";
            this.WindowState = System.Windows.Forms.FormWindowState.Maximized;
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private Tao.Platform.Windows.SimpleOpenGlControl GLC;
        private System.IO.Ports.SerialPort Hand;
        private System.Windows.Forms.Label accelX;
        private System.Windows.Forms.Label accelY;
        private System.Windows.Forms.Label magX;
        private System.Windows.Forms.Label accelZ;
        private System.Windows.Forms.Label magZ;
        private System.Windows.Forms.Label magY;
        private System.Windows.Forms.Label mouseY;
        private System.Windows.Forms.Label mouseX;
        private System.Windows.Forms.Label sensor0;
        private System.Windows.Forms.Label sensor1;
        private System.Windows.Forms.Label sensor3;
        private System.Windows.Forms.Label sensor2;
        private System.Windows.Forms.Label sensor5;
        private System.Windows.Forms.Label sensor4;
        private System.Windows.Forms.Label sensor7;
        private System.Windows.Forms.Label sensor6;
        private System.Windows.Forms.Label sensor9;
        private System.Windows.Forms.Label sensor8;
    }
}

