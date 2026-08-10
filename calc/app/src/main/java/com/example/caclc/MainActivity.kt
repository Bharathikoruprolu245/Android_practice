package com.example.caclc

import android.content.IntentFilter
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.os.unregisterForAllProfilingResults
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat

class MainActivity : AppCompatActivity() {


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)

        // Apply window insets to the root layout
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }

        Log.d("SimpleCalculatorLog", "onCreate() function called")

        val num1EditText: EditText = findViewById(R.id.num1Text)
        val num2EditText: EditText = findViewById(R.id.num2Text)
        val resTextView: TextView = findViewById(R.id.resText)

        val addBtn = findViewById<Button>(R.id.addBtn)
        val subBtn = findViewById<Button>(R.id.subBtn)
        val mulBtn = findViewById<Button>(R.id.mulBtn)
        val divBtn = findViewById<Button>(R.id.divBtn)

        val buttons = listOf(addBtn to "+", subBtn to "-", mulBtn to "*", divBtn to "/")

        buttons.forEach { (btn, op) ->
            btn.setOnClickListener {
                val n1 = num1EditText.text.toString()
                val n2 = num2EditText.text.toString()
                resTextView.text = calcVal(n1, n2, op)
            }
        }
    }

    private fun calcVal(num1Text: String, num2Text: String, op: String): String {
        val n1 = num1Text.toDoubleOrNull()
        val n2 = num2Text.toDoubleOrNull()

        if (n1 == null || n2 == null) return "Invalid Input"

        return when (op) {
            "+" -> (n1 + n2).toString()
            "-" -> (n1 - n2).toString()
            "*" -> (n1 * n2).toString()
            "/" -> if (n2 != 0.0) (n1 / n2).toString() else "Error: Div by 0"
            else -> "Unknown Op"
        }
    }
}
