package com.mariofronza.maskdetector

import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.view.View
import android.widget.Toast
import kotlinx.android.synthetic.main.activity_main.*

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
    }

    fun goToFdActivity(view: View) {
        val quantity = editTextNumber.text.toString()
        if (quantity == "") {
            Toast.makeText(this, "Informe uma quantidade", Toast.LENGTH_SHORT)
        }

        Intent(this, FdActivity::class.java).apply {
            putExtra("quantity", quantity)
            startActivity(this)
        }
    }
}