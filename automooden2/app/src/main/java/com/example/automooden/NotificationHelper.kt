package com.example.automooden

import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.Context
import android.os.Build

import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat

object NotificationHelper {

    private const val CHANNEL_ID =
        "danger_driver"

    fun createChannel(context: Context) {

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {

            val channel = NotificationChannel(

                CHANNEL_ID,

                "Опасные водители",

                NotificationManager.IMPORTANCE_HIGH
            )

            val manager =
                context.getSystemService(
                    NotificationManager::class.java
                )

            manager.createNotificationChannel(channel)
        }
    }

    fun showDangerNotification(
        context: Context,
        number: String
    ) {

        val builder =
            NotificationCompat.Builder(
                context,
                CHANNEL_ID
            )

                .setSmallIcon(
                    android.R.drawable.ic_dialog_alert
                )

                .setContentTitle(
                    "Опасный водитель"
                )

                .setContentText(
                    "Номер $number имеет рейтинг ниже 3"
                )

                .setPriority(
                    NotificationCompat.PRIORITY_HIGH
                )

        NotificationManagerCompat
            .from(context)
            .notify(1, builder.build())
    }
}